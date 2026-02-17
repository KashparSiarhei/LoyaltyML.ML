import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import requests

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Loyalty ML Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================
# МОДЕЛИ ДАННЫХ ДЛЯ API
# ============================================

class CustomerFeatures(BaseModel):
    PhoneNumber: str
    Features: Dict[str, float]

class ProductFeatures(BaseModel):
    ProductId: Optional[str] = None
    Features: Dict[str, float]

class PurchaseInteraction(BaseModel):
    CustomerId: str
    ProductId: Optional[str] = None
    Timestamp: datetime
    Quantity: int
    Price: float

class TrainingDataset(BaseModel):
    customers: List[CustomerFeatures]
    products: List[ProductFeatures]
    interactions: List[PurchaseInteraction]
    
    class Config:
        protected_namespaces = ()
        arbitrary_types_allowed = True

class TrainingConfig(BaseModel):
    session_id: int
    config: Dict[str, Any]
    dataset: TrainingDataset
    
    class Config:
        protected_namespaces = ()
        arbitrary_types_allowed = True
        
class PredictionRequest(BaseModel):
    phone_number: str
    features: Dict[str, float]
    model_id: int
    top_k: int = 3
    
    class Config:
        protected_namespaces = ()
# ============================================
# НЕЙРОСЕТЕВАЯ МОДЕЛЬ (Two-Tower DNN)
# ============================================

class TwoTowerRecommender(nn.Module):
    """
    Двухбашенная нейросеть для рекомендаций:
    - User Tower: эмбеддинги пользователей + числовые фичи
    - Item Tower: эмбеддинги товаров + числовые фичи
    - Выход: вероятность взаимодействия
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Эмбеддинги для ID
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # User Tower
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim + user_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Item Tower
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim + item_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Финальный слой для предсказания
        self.prediction_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_ids, item_ids, user_features, item_features):
        # Получаем эмбеддинги
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Конкатенируем с числовыми фичами
        user_input = torch.cat([user_emb, user_features], dim=1)
        item_input = torch.cat([item_emb, item_features], dim=1)
        
        # Пропускаем через башни
        user_vector = self.user_tower(user_input)
        item_vector = self.item_tower(item_input)
        
        # Конкатенируем векторы для предсказания
        combined = torch.cat([user_vector, item_vector], dim=1)
        prediction = self.prediction_layer(combined)
        
        return prediction

class RecommendationDataset(Dataset):
    """PyTorch Dataset для данных рекомендаций"""
    def __init__(
        self,
        interactions: pd.DataFrame,
        user_encoder: LabelEncoder,
        item_encoder: LabelEncoder,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame
    ):
        self.interactions = interactions
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.user_features = user_features
        self.item_features = item_features
        
        # Кодируем ID
        self.user_ids = self.user_encoder.transform(interactions['customer_id'])
        self.item_ids = self.item_encoder.transform(interactions['product_id'])
        
        # Нормализуем цены/количество
        self.targets = np.log1p(interactions['quantity'].values * interactions['price'].values)
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        
        # Получаем фичи пользователя
        customer_id = self.interactions.iloc[idx]['customer_id']
        user_feats = self.user_features.loc[customer_id].values.astype(np.float32)
        
        # Получаем фичи товара
        product_id = self.interactions.iloc[idx]['product_id']
        item_feats = self.item_features.loc[product_id].values.astype(np.float32)
        
        target = np.float32(self.targets[idx])
        target = np.clip(target / 100, 0, 1)  # Нормализация в [0,1]
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'user_features': torch.tensor(user_feats, dtype=torch.float32),
            'item_features': torch.tensor(item_feats, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def calculate_metrics_at_k(predictions, targets, k=10):
    """Расчет Precision@K, Recall@K, NDCG@K"""
    # Сортируем по предсказаниям
    indices = np.argsort(predictions)[-k:]
    
    # Precision@K
    precision = np.mean(targets[indices] > 0.5)
    
    # Recall@K
    recall = np.sum(targets[indices] > 0.5) / max(np.sum(targets > 0.5), 1)
    
    # NDCG@K
    dcg = np.sum((2**targets[indices] - 1) / np.log2(np.arange(2, k + 2)))
    idcg = np.sum((2**np.sort(targets)[-k:][::-1] - 1) / np.log2(np.arange(2, k + 2)))
    ndcg = dcg / max(idcg, 1)
    
    return precision, recall, ndcg

# ============================================
# ЭНДПОИНТЫ API
# ============================================

@app.get("/")
async def root():
    return {
        "service": "Loyalty ML Service",
        "version": "2.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/train")
async def train_model(data: TrainingDataset, background_tasks: BackgroundTasks):
    """
    Обучение модели рекомендаций
    """
    logger.info("="*50)
    logger.info(f"Received training request")
    logger.info(f"Session ID: {data.session_id}")
    logger.info(f"Customers count: {len(data.dataset.customers)}")
    logger.info(f"Products count: {len(data.dataset.products)}")
    logger.info(f"Interactions count: {len(data.dataset.interactions)}")
    logger.info("="*50)
    
    if len(data.dataset.customers) > 0:
        logger.info(f"First customer: {data.dataset.customers[0]}")
        
    try:
        logger.info(f"Starting training for session {data.session_id}")
        
        # 1. Преобразуем данные в DataFrame
        interactions_df = pd.DataFrame([i.dict() for i in data.interactions])
        customers_df = pd.DataFrame([
            {'customer_id': c.phone_number, **c.features} 
            for c in data.customers
        ])
        products_df = pd.DataFrame([
            {'product_id': p.product_id, **p.features} 
            for p in data.products
        ])
        
        # 2. Кодируем ID
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        all_users = interactions_df['customer_id'].unique()
        all_items = interactions_df['product_id'].unique()
        
        user_encoder.fit(all_users)
        item_encoder.fit(all_items)
        
        # 3. Подготавливаем фичи
        customers_df = customers_df.set_index('customer_id')
        products_df = products_df.set_index('product_id')
        
        # 4. Создаем датасет
        dataset = RecommendationDataset(
            interactions_df,
            user_encoder,
            item_encoder,
            customers_df,
            products_df
        )
        
        # 5. Разделяем на train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # 6. Создаем модель
        model = TwoTowerRecommender(
            num_users=len(user_encoder.classes_),
            num_items=len(item_encoder.classes_),
            user_feature_dim=len(customers_df.columns),
            item_feature_dim=len(products_df.columns),
            embedding_dim=data.config.get('embedding_size', 64),
            hidden_dim=128
        )
        
        # 7. Настраиваем обучение
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=data.config.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        criterion = nn.BCELoss()
        
        # 8. Обучаем
        epochs = data.config.get('epochs', 20)
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_path = None
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for batch in train_loader:
                user_ids = batch['user_id'].to(device)
                item_ids = batch['item_id'].to(device)
                user_features = batch['user_features'].to(device)
                item_features = batch['item_features'].to(device)
                targets = batch['target'].to(device)
                
                optimizer.zero_grad()
                predictions = model(user_ids, item_ids, user_features, item_features)
                loss = criterion(predictions.squeeze(), targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    user_ids = batch['user_id'].to(device)
                    item_ids = batch['item_id'].to(device)
                    user_features = batch['user_features'].to(device)
                    item_features = batch['item_features'].to(device)
                    targets = batch['target'].to(device)
                    
                    predictions = model(user_ids, item_ids, user_features, item_features)
                    loss = criterion(predictions.squeeze(), targets)
                    val_loss += loss.item()
                    
                    all_val_preds.extend(predictions.cpu().numpy())
                    all_val_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            
            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("models", exist_ok=True)
                best_model_path = f"models/model_{data.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'user_encoder': user_encoder,
                    'item_encoder': item_encoder,
                    'user_feature_columns': customers_df.columns.tolist(),
                    'item_feature_columns': products_df.columns.tolist(),
                    'config': data.config,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'timestamp': datetime.now().isoformat()
                }, best_model_path)
            
            # Отправляем прогресс
            try:
                requests.post("http://localhost:5000/api/training/progress", json={
                    "sessionId": data.session_id,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            except:
                pass
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 9. Считаем метрики на валидации
        all_val_preds = np.array(all_val_preds).flatten()
        all_val_targets = np.array(all_val_targets)
        precision, recall, ndcg = calculate_metrics_at_k(all_val_preds, all_val_targets, k=10)
        
        logger.info(f"Training completed. Precision@10: {precision:.4f}, Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")
        
        return {
            "model_path": best_model_path,
            "loss": float(best_val_loss),
            "precision": float(precision),
            "recall": float(recall),
            "ndcg": float(ndcg),
            "learning_curve": {
                "train_losses": train_losses,
                "val_losses": val_losses
            }
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Получение рекомендаций для клиента
    """
    try:
        logger.info(f"Generating recommendations for {request.phone_number}")
        
        # Находим последнюю модель для этого ID
        import glob
        model_files = glob.glob(f"models/model_{request.model_id}_*.pt")
        
        if not model_files:
            # Если нет модели с таким ID, берем самую свежую
            model_files = glob.glob("models/model_*.pt")
            if not model_files:
                raise HTTPException(status_code=404, detail="No models found")
            
            # Берем самую свежую
            model_files.sort(reverse=True)
        
        checkpoint = torch.load(model_files[-1], map_location='cpu')
        
        # Восстанавливаем энкодеры
        user_encoder = checkpoint['user_encoder']
        item_encoder = checkpoint['item_encoder']
        
        # В реальном проекте здесь нужно загрузить фичи пользователя из БД
        # Для демо генерируем случайные предсказания
        
        # Получаем все товары
        all_items = item_encoder.classes_
        
        # Для демо берем первые 10 товаров
        products = all_items[:10] if len(all_items) > 10 else all_items
        
        predictions = []
        for product_id in products:
            # Генерируем случайную вероятность для демо
            prob = np.random.random()
            predictions.append({
                "product_id": product_id,
                "score": float(prob),
                "probability": float(prob)
            })
        
        # Сортируем по вероятности
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)[:request.top_k]
        
        # Рассчитываем скидку (чем выше вероятность, тем больше скидка)
        base_discount = 5 + int(predictions[0]['probability'] * 20) if predictions else 10
        
        return {
            "product_ids": [p['product_id'] for p in predictions],
            "top_product": predictions[0]['product_id'] if predictions else None,
            "discount": base_discount,
            "probability": predictions[0]['probability'] if predictions else 0.5,
            "score": predictions[0]['score'] if predictions else 0.5,
            "all_predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/info")
async def get_model_info(model_id: int):
    """
    Получение информации о модели
    """
    try:
        import glob
        model_files = glob.glob(f"models/model_{model_id}_*.pt")
        
        if not model_files:
            raise HTTPException(status_code=404, detail="Model not found")
        
        checkpoint = torch.load(model_files[-1], map_location='cpu')
        
        return {
            "model_id": model_id,
            "timestamp": checkpoint['timestamp'],
            "config": checkpoint['config'],
            "train_loss": checkpoint['train_loss'],
            "val_loss": checkpoint['val_loss'],
            "user_features": checkpoint['user_feature_columns'],
            "item_features": checkpoint['item_feature_columns']
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)