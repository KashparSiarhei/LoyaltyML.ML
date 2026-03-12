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
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import logging
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import asyncio
from contextlib import asynccontextmanager
import aiohttp
import gc
import signal
import glob
import re
from collections import defaultdict
import warnings
import pickle
import hashlib
import psutil
import joblib
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Импортируем наши модули
from annoy_index import annoy_index_manager, AnnoyIndexManager
from feature_store import EnhancedAsyncFeatureStore, MissingItemStrategy

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_MODEL_CACHE_SIZE = int(os.getenv("MAX_MODEL_CACHE_SIZE", "10"))
PREDICTION_TTL = int(os.getenv("PREDICTION_TTL", "300"))
MODELS_DIR = os.getenv("MODELS_DIR", "models")
INDEXES_DIR = os.getenv("INDEXES_DIR", "indexes")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
INCREMENTAL_LEARNING_RATE = float(os.getenv("INCREMENTAL_LEARNING_RATE", "0.0005"))
INCREMENTAL_EPOCHS = int(os.getenv("INCREMENTAL_EPOCHS", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MODEL_TTL = int(os.getenv("MODEL_TTL", "7200"))  # 2 часа
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 час
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# Multiprocess Prometheus
if 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
    os.environ['PROMETHEUS_MULTIPROC_DIR'] = os.path.join(os.path.dirname(__file__), 'prometheus_data')
    os.makedirs(os.environ['PROMETHEUS_MULTIPROC_DIR'], exist_ok=True)

# Логирование
logging.basicConfig(
    level=logging.INFO if ENVIRONMENT == "production" else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_service.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Метрики Prometheus
try:
    PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions', ['model_id', 'status'])
    PREDICTION_FALLBACK = Counter('predictions_fallback_total', 'Fallback predictions', ['model_id', 'reason'])
    TRAINING_COUNTER = Counter('training_total', 'Total training runs', ['status'])
    INCREMENTAL_TRAINING_COUNTER = Counter('incremental_training_total', 'Total incremental training runs', ['status'])
    PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency', ['model_id'])
    MODEL_VERSION_GAUGE = Gauge('model_version', 'Current model version', ['model_id', 'metric'])
    CACHE_HITS = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
    CACHE_MISSES = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
    ACTIVE_REQUESTS = Gauge('active_requests', 'Active requests')
    MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
    CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percent')
except ValueError as e:
    logger.warning(f"Metrics already registered: {e}")
    registry = REGISTRY
    PREDICTION_COUNTER = registry._names_to_collectors.get('predictions_total')
    PREDICTION_FALLBACK = registry._names_to_collectors.get('predictions_fallback_total')
    TRAINING_COUNTER = registry._names_to_collectors.get('training_total')
    INCREMENTAL_TRAINING_COUNTER = registry._names_to_collectors.get('incremental_training_total')
    PREDICTION_LATENCY = registry._names_to_collectors.get('prediction_latency_seconds')
    MODEL_VERSION_GAUGE = registry._names_to_collectors.get('model_version')
    CACHE_HITS = registry._names_to_collectors.get('cache_hits_total')
    CACHE_MISSES = registry._names_to_collectors.get('cache_misses_total')
    ACTIVE_REQUESTS = registry._names_to_collectors.get('active_requests')
    MEMORY_USAGE = registry._names_to_collectors.get('memory_usage_bytes')
    CPU_USAGE = registry._names_to_collectors.get('cpu_usage_percent')

redis_client = None
feature_store = None
active_requests = 0
request_lock = asyncio.Lock()
background_tasks = set()
model_cache = None
annoy_index_manager = AnnoyIndexManager()


# ============================================
# PYDANTIC МОДЕЛИ
# ============================================

class CustomerFeatures(BaseModel):
    PhoneNumber: str
    Features: Dict[str, float]
    
    @field_validator('PhoneNumber')
    def validate_phone(cls, v):
        if not v or len(v) < 5:
            raise ValueError('Invalid phone number')
        return v
    
    class Config:
        protected_namespaces = ()

class ProductFeatures(BaseModel):
    ProductId: Optional[str] = None
    Features: Dict[str, float]
    
    class Config:
        protected_namespaces = ()

class PurchaseInteraction(BaseModel):
    CustomerId: str
    ProductId: Optional[str] = None
    Timestamp: datetime
    Quantity: int = Field(ge=0, le=1000)
    Price: float = Field(ge=0, le=1000000)
    
    class Config:
        protected_namespaces = ()

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

# ========== ИСПРАВЛЕНИЕ: Добавлено поле new_model_id ==========
class IncrementalTrainingRequest(BaseModel):
    base_model_id: int
    new_model_id: int  # ← Добавлено!
    new_interactions: List[Dict[str, Any]]
    training_params: Optional[Dict[str, Any]] = {}
    
    class Config:
        protected_namespaces = ()
# ============================================================

class IncrementalTrainingResponse(BaseModel):
    success: bool
    metrics: Dict[str, float]
    model_path: str
    error: Optional[str] = None
    
    class Config:
        protected_namespaces = ()

class PredictionRequest(BaseModel):
    phone_number: str
    features: Dict[str, float]
    model_id: int
    top_k: int = Field(default=3, ge=1, le=20)
    explain: bool = False
    
    class Config:
        protected_namespaces = ()

class PredictionResponse(BaseModel):
    product_ids: List[str]
    top_product: Optional[str]
    discount: float
    probability: float
    score: float
    all_predictions: List[Dict[str, float]]
    explanation: Optional[Dict[str, float]] = None
    model_version: str
    cached: bool = False
    fallback: bool = False
    
    class Config:
        protected_namespaces = ()

class ModelInfo(BaseModel):
    model_id: int
    version: str
    created_at: datetime
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    config: Dict[str, Any]
    
    class Config:
        protected_namespaces = ()

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    device: str
    redis: str
    models_loaded: int
    gpu_available: bool
    gpu_memory: Optional[int] = None
    active_requests: int = 0
    indexes_loaded: int
    memory_usage: float
    cpu_usage: float
    uptime: str
    
    class Config:
        protected_namespaces = ()


# ============================================
# НЕЙРОСЕТЬ
# ============================================

class ImprovedTwoTowerRecommender(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.user_embedding_norm = nn.LayerNorm(embedding_dim)
        self.item_embedding_norm = nn.LayerNorm(embedding_dim)
        
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim + user_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim + item_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(self, user_ids, item_ids, user_features, item_features):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        user_emb = self.user_embedding_norm(user_emb)
        item_emb = self.item_embedding_norm(item_emb)
        
        user_input = torch.cat([user_emb, user_features], dim=1)
        item_input = torch.cat([item_emb, item_features], dim=1)
        
        user_vector = self.user_tower(user_input).unsqueeze(1)
        item_vector = self.item_tower(item_input).unsqueeze(1)
        
        attended_user, _ = self.cross_attention(user_vector, item_vector, item_vector)
        attended_item, _ = self.cross_attention(item_vector, user_vector, user_vector)
        
        combined = torch.cat([
            user_vector.squeeze(1),
            item_vector.squeeze(1),
            attended_user.squeeze(1) + attended_item.squeeze(1)
        ], dim=1)
        
        logits = self.classifier(combined).squeeze()
        return torch.sigmoid(logits / self.temperature)
    
    def get_user_embedding(self, user_id, user_features):
        user_emb = self.user_embedding(user_id)
        user_emb = self.user_embedding_norm(user_emb)
        user_input = torch.cat([user_emb, user_features], dim=1)
        user_vector = self.user_tower(user_input)
        return user_vector
    
    def get_item_embedding(self, item_id, item_features):
        item_emb = self.item_embedding(item_id)
        item_emb = self.item_embedding_norm(item_emb)
        item_input = torch.cat([item_emb, item_features], dim=1)
        item_vector = self.item_tower(item_input)
        return item_vector
    
    def get_config(self):
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'user_feature_dim': self.user_feature_dim,
            'item_feature_dim': self.item_feature_dim,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        }


# ============================================
# ДАТАСЕТЫ
# ============================================

class RecommendationDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        user_encoder: LabelEncoder,
        item_encoder: LabelEncoder,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        augment: bool = False
    ):
        self.interactions = interactions
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.user_features = user_features
        self.item_features = item_features
        self.augment = augment
        self.feature_columns = item_features.columns.tolist()
        
        # Фильтруем только валидные взаимодействия
        valid_indices = []
        user_ids = []
        item_ids = []
        
        for idx, row in interactions.iterrows():
            try:
                user_id = row['customer_id']
                item_id = row['product_id']
                
                if user_id in user_features.index and item_id in item_features.index:
                    user_idx = user_encoder.transform([user_id])[0]
                    item_idx = item_encoder.transform([item_id])[0]
                    user_ids.append(user_idx)
                    item_ids.append(item_idx)
                    valid_indices.append(idx)
            except:
                continue
        
        self.user_ids = np.array(user_ids)
        self.item_ids = np.array(item_ids)
        self.valid_indices = valid_indices
        
        # Вычисляем таргеты
        if len(valid_indices) > 0:
            values = interactions.iloc[valid_indices]['quantity'].values * interactions.iloc[valid_indices]['price'].values
            self.targets = np.log1p(values)
            self.targets = np.clip(self.targets / (self.targets.max() + 1e-8), 0, 1)
        else:
            self.targets = np.array([])
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        
        actual_idx = self.valid_indices[idx]
        customer_id = self.interactions.iloc[actual_idx]['customer_id']
        product_id = self.interactions.iloc[actual_idx]['product_id']
        
        user_feats = self.user_features.loc[customer_id].values.astype(np.float32)
        item_feats = self.item_features.loc[product_id].values.astype(np.float32)
        
        target = np.float32(self.targets[idx])
        
        if self.augment and np.random.random() < 0.3:
            user_feats += np.random.normal(0, 0.01, user_feats.shape)
            item_feats += np.random.normal(0, 0.01, item_feats.shape)
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'user_features': torch.tensor(user_feats, dtype=torch.float32),
            'item_features': torch.tensor(item_feats, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }


class IncrementalDataset(Dataset):
    def __init__(
        self,
        interactions: List[Dict[str, Any]],
        user_encoder: LabelEncoder,
        item_encoder: LabelEncoder,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        augment: bool = False
    ):
        self.interactions = interactions
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.user_features = user_features
        self.item_features = item_features
        self.augment = augment
        self.user_feature_columns = user_features.columns.tolist() if user_features is not None and not user_features.empty else []
        self.item_feature_columns = item_features.columns.tolist() if item_features is not None and not item_features.empty else []
        
        self.df = pd.DataFrame(interactions)
        
        # Фильтруем валидные
        valid_indices = []
        user_ids = []
        item_ids = []

        user_classes = set(map(str, getattr(user_encoder, "classes_", [])))
        item_classes = set(map(str, getattr(item_encoder, "classes_", [])))
        
        for idx, row in self.df.iterrows():
            try:
                customer_id = str(row.get('customer_id') or row.get('CustomerId', ''))
                product_id = str(row.get('product_id') or row.get('ProductId', ''))
                
                if not customer_id or not product_id:
                    continue
                    
                # Энкодеры: если ID не известен базовой модели — используем индекс 0 (unknown).
                # Это позволяет онлайн-дообучению работать на новых user/item, не выбрасывая весь батч.
                if customer_id in user_classes:
                    user_idx = int(user_encoder.transform([customer_id])[0])
                else:
                    user_idx = 0

                if product_id in item_classes:
                    item_idx = int(item_encoder.transform([product_id])[0])
                else:
                    item_idx = 0

                user_ids.append(user_idx)
                item_ids.append(item_idx)
                valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"Skipping interaction: {e}")
                continue
        
        self.user_ids = np.array(user_ids) if user_ids else np.array([])
        self.item_ids = np.array(item_ids) if item_ids else np.array([])
        self.valid_indices = valid_indices
        
        if len(valid_indices) > 0:
            quantities = []
            prices = []
            
            for idx in valid_indices:
                row = self.df.iloc[idx]
                qty = row.get('quantity')
                if qty is None:
                    qty = row.get('Quantity')
                if qty is None:
                    qty = 1
                
                prc = row.get('price')
                if prc is None:
                    prc = row.get('Price')
                if prc is None:
                    prc = 0
                
                quantities.append(float(qty))
                prices.append(float(prc))
            
            quantities = np.array(quantities)
            prices = np.array(prices)
            
            values = quantities * prices
            self.targets = np.log1p(values)
            if len(self.targets) > 0 and self.targets.max() > 0:
                self.targets = np.clip(self.targets / (self.targets.max() + 1e-8), 0, 1)
            else:
                self.targets = np.zeros_like(self.targets)
        else:
            self.targets = np.array([])
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        
        customer_id = str(row.get('customer_id') or row.get('CustomerId'))
        product_id = str(row.get('product_id') or row.get('ProductId'))
        
        if self.user_features is None or self.user_features.empty or customer_id not in self.user_features.index:
            user_feats = np.zeros(len(self.user_feature_columns) if self.user_feature_columns else 1, dtype=np.float32)
        else:
            user_feats = self.user_features.loc[customer_id].values.astype(np.float32)
        
        if self.item_features is None or self.item_features.empty or product_id not in self.item_features.index:
            item_feats = np.zeros(len(self.item_feature_columns) if self.item_feature_columns else 1, dtype=np.float32)
        else:
            item_feats = self.item_features.loc[product_id].values.astype(np.float32)
        
        target = np.float32(self.targets[idx])
        
        if self.augment and np.random.random() < 0.2:
            user_feats += np.random.normal(0, 0.005, user_feats.shape)
            item_feats += np.random.normal(0, 0.005, item_feats.shape)
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'user_features': torch.tensor(user_feats, dtype=torch.float32),
            'item_features': torch.tensor(item_feats, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }


# ============================================
# МЕТРИКИ
# ============================================

def calculate_metrics(predictions, targets, k_list=[1, 3, 5, 10]):
    metrics = {}
    predictions = np.array(predictions).flatten()
    targets = np.array(targets)
    
    for k in k_list:
        if len(predictions) >= k:
            indices = np.argsort(predictions)[-k:]
            metrics[f'precision@{k}'] = float(np.mean(targets[indices] > 0.5))
            recall = np.sum(targets[indices] > 0.5) / max(np.sum(targets > 0.5), 1)
            metrics[f'recall@{k}'] = float(recall)
            
            dcg = np.sum((2**targets[indices] - 1) / np.log2(np.arange(2, k + 2)))
            idcg = np.sum((2**np.sort(targets)[-k:][::-1] - 1) / np.log2(np.arange(2, k + 2)))
            metrics[f'ndcg@{k}'] = float(dcg / max(idcg, 1))
    
    try:
        metrics['auc'] = float(roc_auc_score(targets > 0.5, predictions))
    except:
        metrics['auc'] = 0.5
    
    try:
        metrics['precision'] = float(precision_score(targets > 0.5, predictions > 0.5, zero_division=0))
        metrics['recall'] = float(recall_score(targets > 0.5, predictions > 0.5, zero_division=0))
    except:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
    
    return metrics


# ============================================
# MODEL CACHE
# ============================================

class ModelCache:
    def __init__(self, max_size=5, ttl=3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.lock = asyncio.Lock()
    
    async def get(self, model_id: int):
        async with self.lock:
            if model_id in self.cache:
                if datetime.now().timestamp() - self.access_times[model_id] < self.ttl:
                    self.access_times[model_id] = datetime.now().timestamp()
                    if CACHE_HITS:
                        CACHE_HITS.labels(cache_type='model').inc()
                    return self.cache[model_id]
                else:
                    del self.cache[model_id]
                    del self.access_times[model_id]
            if CACHE_MISSES:
                CACHE_MISSES.labels(cache_type='model').inc()
            return None
    
    async def set(self, model_id: int, model_data: dict):
        async with self.lock:
            if len(self.cache) >= self.max_size:
                oldest = min(self.access_times.keys(), key=lambda x: self.access_times[x])
                del self.cache[oldest]
                del self.access_times[oldest]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.cache[model_id] = model_data
            self.access_times[model_id] = datetime.now().timestamp()


# ============================================
# ОБУЧЕНИЕ
# ============================================

async def train_model_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    session_id: int
) -> dict:
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    criterion = nn.BCELoss()
    
    epochs = config.get('epochs', 50)
    patience = config.get('patience', 10)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_metrics = {}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            user_features = batch['user_features'].to(device)
            item_features = batch['item_features'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids, user_features, item_features)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
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
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_batches += 1
                
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
        
        val_loss /= val_batches
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = calculate_metrics(all_val_preds, all_val_targets)
            logger.info(f"New best model at epoch {epoch+1} with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return {
        'model': model, 
        'best_val_loss': best_val_loss,
        'metrics': best_metrics
    }


# ========== ИСПРАВЛЕНИЕ: Добавлен параметр new_model_id ==========
async def incremental_train(
    training_id: int,
    base_model_id: int,
    new_model_id: int,  # ← Добавлено!
    new_interactions: List[Dict[str, Any]],
    training_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Инкрементальное дообучение модели на новых данных
    """
    start_time = datetime.now()
    
    try:
        if INCREMENTAL_TRAINING_COUNTER:
            INCREMENTAL_TRAINING_COUNTER.labels(status='started').inc()
        
        logger.info("="*60)
        logger.info(f"Incremental training for base model {base_model_id}")
        logger.info(f"New model will have ID: {new_model_id}")  # ← Логируем
        logger.info(f"New interactions: {len(new_interactions)}")
        logger.info("="*60)
        
        # Находим файл базовой модели
        model_files = glob.glob(os.path.join(MODELS_DIR, f"model_{base_model_id}_*.pt"))
        if not model_files:
            error_msg = f"Base model {base_model_id} not found"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'metrics': {},
                'model_path': ''
            }
        
        model_files.sort(reverse=True)
        base_model_path = model_files[0]
        logger.info(f"Loading base model from {base_model_path}")
        
        # Загружаем чекпоинт базовой модели
        checkpoint = torch.load(base_model_path, map_location='cpu')
        
        # Извлекаем все необходимые компоненты
        user_encoder = checkpoint['user_encoder']
        item_encoder = checkpoint['item_encoder']
        user_scaler = checkpoint.get('user_scaler')
        item_scaler = checkpoint.get('item_scaler')
        user_feature_columns = checkpoint['user_feature_columns']
        item_feature_columns = checkpoint['item_feature_columns']
        base_config = checkpoint['config']
        
        # Определяем устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Создаем модель с той же архитектурой
        model = ImprovedTwoTowerRecommender(
            num_users=len(user_encoder.classes_),
            num_items=len(item_encoder.classes_),
            user_feature_dim=len(user_feature_columns),
            item_feature_dim=len(item_feature_columns),
            embedding_dim=base_config.get('embedding_size', 64),
            hidden_dim=base_config.get('hidden_size', 256),
            dropout=base_config.get('dropout', 0.3)
        ).to(device)
        
        # Загружаем веса
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        
        # Получаем уникальных пользователей и товары из новых данных
        unique_users = set()
        unique_items = set()
        
        for interaction in new_interactions:
            customer_id = str(interaction.get('customer_id') or interaction.get('CustomerId', ''))
            product_id = str(interaction.get('product_id') or interaction.get('ProductId', ''))
            
            if customer_id:
                unique_users.add(customer_id)
            if product_id:
                unique_items.add(product_id)
        
        logger.info(f"Found {len(unique_users)} unique users and {len(unique_items)} unique items in new data")
        
        # Получаем фичи пользователей из базы через feature_store
        user_features_dict = {}
        if feature_store:
            for user_id in unique_users:
                try:
                    features = await feature_store.get_user_features(user_id)
                    if features:
                        user_features_dict[user_id] = features
                except Exception as e:
                    logger.warning(f"Error getting features for user {user_id}: {e}")
        
        # Создаем DataFrame для фичей пользователей
        user_data = []
        for user_id in unique_users:
            if user_id in user_features_dict:
                features = user_features_dict[user_id]
            else:
                # Используем дефолтные значения
                features = {col: 0.0 for col in user_feature_columns}
            
            features['customer_id'] = user_id
            user_data.append(features)
        
        if user_data:
            users_df = pd.DataFrame(user_data).set_index('customer_id')
            for col in user_feature_columns:
                if col not in users_df.columns:
                    users_df[col] = 0.0
            users_df = users_df[user_feature_columns]
        else:
            users_df = pd.DataFrame(columns=user_feature_columns)
        
        # Получаем фичи товаров
        item_features_dict = {}
        if feature_store:
            for item_id in unique_items:
                try:
                    features = await feature_store.get_item_features(item_id)
                    if features:
                        item_features_dict[item_id] = features
                except Exception as e:
                    logger.warning(f"Error getting features for item {item_id}: {e}")
        
        # Создаем DataFrame для фичей товаров
        item_data = []
        for item_id in unique_items:
            if item_id in item_features_dict:
                features = item_features_dict[item_id]
            else:
                # Используем умные дефолты
                features = MissingItemStrategy.get_smart_defaults(item_feature_columns)
            
            features['product_id'] = item_id
            item_data.append(features)
        
        if item_data:
            items_df = pd.DataFrame(item_data).set_index('product_id')
            for col in item_feature_columns:
                if col not in items_df.columns:
                    items_df[col] = 0.0
            items_df = items_df[item_feature_columns]
        else:
            items_df = pd.DataFrame(columns=item_feature_columns)
        
        if users_df.empty:
            logger.warning("User features DataFrame is empty, creating dummy features")
            dummy_data = []
            for user_id in unique_users:
                dummy_row = {col: 0.0 for col in user_feature_columns}
                dummy_row['customer_id'] = user_id
                dummy_data.append(dummy_row)
            users_df = pd.DataFrame(dummy_data).set_index('customer_id')

        if items_df.empty:
            logger.warning("Item features DataFrame is empty, creating dummy features")
            dummy_data = []
            for item_id in unique_items:
                dummy_row = {col: 0.0 for col in item_feature_columns}
                dummy_row['product_id'] = item_id
                dummy_data.append(dummy_row)
            items_df = pd.DataFrame(dummy_data).set_index('product_id')
        
        # Нормализуем фичи
        if user_scaler and len(users_df) > 0:
            users_df_scaled = pd.DataFrame(
                user_scaler.transform(users_df),
                index=users_df.index,
                columns=user_feature_columns
            )
        else:
            users_df_scaled = users_df
        
        if item_scaler and len(items_df) > 0:
            items_df_scaled = pd.DataFrame(
                item_scaler.transform(items_df),
                index=items_df.index,
                columns=item_feature_columns
            )
        else:
            items_df_scaled = items_df
        
        # Создаем Dataset для инкрементального обучения
        train_dataset = IncrementalDataset(
            interactions=new_interactions,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            user_features=users_df_scaled,
            item_features=items_df_scaled,
            augment=True
        )
        
        if len(train_dataset) == 0:
            error_msg = "No valid interactions for incremental training"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'metrics': {},
                'model_path': ''
            }
        
        if len(train_dataset) < 5:
            logger.warning(f"Very small dataset: {len(train_dataset)} samples. Training may not be effective.")
    
        logger.info(f"Created dataset with {len(train_dataset)} valid samples")
        
        # Нормализация параметров из бэка (camelCase → snake_case), чтобы UI мог слать learningRate/batchSize
        normalized_params = dict(training_params or {})
        if 'learningRate' in normalized_params and 'learning_rate' not in normalized_params:
            normalized_params['learning_rate'] = normalized_params['learningRate']
        if 'batchSize' in normalized_params and 'batch_size' not in normalized_params:
            normalized_params['batch_size'] = normalized_params['batchSize']

        # Создаем DataLoader с проверкой на пустой dataset
        if len(train_dataset) > 0:
            batch_size = min(int(normalized_params.get('batch_size', 128)), len(train_dataset))
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                num_workers=0
            )
        else:
            error_msg = "No valid samples after filtering"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'metrics': {},
                'model_path': ''
            }
        
        # Настройки оптимизатора
        learning_rate = float(normalized_params.get('learning_rate', INCREMENTAL_LEARNING_RATE))
        epochs = int(normalized_params.get('epochs', INCREMENTAL_EPOCHS))
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=base_config.get('weight_decay', 1e-5)
        )
        
        criterion = nn.BCELoss()
        
        # Инкрементальное обучение
        logger.info(f"Starting incremental training for {epochs} epochs")
        
        model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            
            for batch in train_loader:
                user_ids = batch['user_id'].to(device)
                item_ids = batch['item_id'].to(device)
                user_features = batch['user_features'].to(device)
                item_features = batch['item_features'].to(device)
                targets = batch['target'].to(device)
                
                optimizer.zero_grad()
                predictions = model(user_ids, item_ids, user_features, item_features)
                loss = criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
            
            avg_loss = epoch_loss / batches if batches > 0 else 0
            epoch_losses.append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Оцениваем модель на новых данных
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in train_loader:
                user_ids = batch['user_id'].to(device)
                item_ids = batch['item_id'].to(device)
                user_features = batch['user_features'].to(device)
                item_features = batch['item_features'].to(device)
                targets = batch['target'].to(device)
                
                predictions = model(user_ids, item_ids, user_features, item_features)
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Вычисляем метрики
        metrics = calculate_metrics(all_preds, all_targets)
        logger.info(f"Post-training metrics: {metrics}")
        
        # ========== ИСПРАВЛЕНИЕ: Используем new_model_id в имени файла ==========
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODELS_DIR, f"model_inc_{base_model_id}_{new_model_id}_{timestamp}.pt")
        # =======================================================================
        
        # Обновляем метрики в чекпоинте
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['metrics'] = metrics
        checkpoint['base_model_id'] = base_model_id
        checkpoint['incremental_training'] = True
        checkpoint['new_interactions_count'] = len(new_interactions)
        checkpoint['incremental_losses'] = epoch_losses
        checkpoint['timestamp'] = timestamp
        checkpoint['version'] = f"{checkpoint.get('version', '3.2.0')}.inc{epochs}"
        checkpoint['training_time_seconds'] = (datetime.now() - start_time).total_seconds()
        
        torch.save(checkpoint, model_path)
        logger.info(f"Incrementally trained model saved to {model_path}")
        
        # Обновляем Annoy индекс с правильным new_model_id
        new_model_id_returned = None
        try:
            new_model_id_returned = await build_annoy_index_for_model(model, item_encoder, item_feature_columns, model_path, device, new_model_id)
        except Exception as e:
            logger.error(f"Failed to build Annoy index for incremental model: {e}")
            
        # Отправляем уведомление в бэкенд
        try:
            async with aiohttp.ClientSession() as session:
                notify_data = {
                    "TrainingId": training_id,
                    "status": "completed",
                    "metrics": metrics,
                    "modelPath": model_path,
                    "embeddingDim": model.embedding_dim,
                    "newModelId": new_model_id  # ← Используем полученный ID
                }
                
                logger.info(f"Sending completion notification for training {training_id} to {BACKEND_URL}/api/OnlineLearning/notify-complete")
                
                async with session.post(
                    f"{BACKEND_URL}/api/OnlineLearning/notify-complete",
                    json=notify_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Successfully notified backend for training {training_id}")
                    else:
                        response_text = await response.text()
                        logger.error(f"❌ Backend returned error {response.status}: {response_text}")
        except Exception as e:
            logger.error(f"Failed to notify backend: {e}")
        
        if INCREMENTAL_TRAINING_COUNTER:
            INCREMENTAL_TRAINING_COUNTER.labels(status='completed').inc()
        
        return {
            'success': True,
            'metrics': metrics,
            'model_path': model_path,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Incremental training failed: {str(e)}", exc_info=True)
        if INCREMENTAL_TRAINING_COUNTER:
            INCREMENTAL_TRAINING_COUNTER.labels(status='failed').inc()
        return {
            'success': False,
            'error': str(e),
            'metrics': {},
            'model_path': ''
        }
# ==================================================================


# ========== ИСПРАВЛЕНИЕ: Добавлен параметр new_model_id ==========
async def build_annoy_index_for_model(model, item_encoder, item_feature_columns, model_path, device, new_model_id=None):
    """Построение Annoy индекса для модели"""
    try:
        model.eval()
        item_embeddings = []
        item_ids_list = []
        
        # Получаем все item_id из энкодера
        all_items = item_encoder.classes_
        
        with torch.no_grad():
            for item_id in all_items:
                try:
                    item_idx = item_encoder.transform([item_id])[0]
                    item_id_str = str(item_id)
                    
                    # Получаем фичи товара
                    if feature_store:
                        item_features_dict = await feature_store.get_item_features(item_id_str)
                        if item_features_dict:
                            item_feats = np.array([item_features_dict.get(col, 0.0) for col in item_feature_columns], dtype=np.float32)
                        else:
                            # Используем дефолтные
                            item_feats = np.zeros(len(item_feature_columns), dtype=np.float32)
                    else:
                        item_feats = np.zeros(len(item_feature_columns), dtype=np.float32)
                    
                    item_tensor = torch.tensor([item_idx], dtype=torch.long).to(device)
                    feat_tensor = torch.tensor([item_feats], dtype=torch.float32).to(device)
                    
                    item_vector = model.get_item_embedding(item_tensor, feat_tensor).cpu().numpy()[0]
                    item_embeddings.append(item_vector)
                    item_ids_list.append(item_id)
                except Exception as e:
                    logger.warning(f"Error getting embedding for item {item_id}: {e}")
                    continue
        
        if item_embeddings:
            # Если new_model_id не передан, пытаемся извлечь из пути
            if new_model_id is None:
                model_id_match = re.search(r'model_(\d+)_', os.path.basename(model_path))
                if model_id_match:
                    new_model_id = int(model_id_match.group(1))
                else:
                    new_model_id = int(datetime.now().timestamp())
                    logger.warning(f"Generated fallback model ID: {new_model_id}")
            
            annoy_index_manager.build_index(
                new_model_id,
                np.array(item_embeddings),
                item_ids_list
            )
            logger.info(f"Built Annoy index for model {new_model_id} with {len(item_embeddings)} items")
            return new_model_id
    except Exception as e:
        logger.error(f"Error building Annoy index: {e}")
        return None
# ==================================================================


# ============================================
# LIFESPAN
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, feature_store, model_cache, annoy_index_manager
    
    start_time = datetime.now()
    
    # Startup
    try:
        redis_client = await redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True,
            max_connections=10, socket_timeout=5, health_check_interval=30,
            retry_on_timeout=True
        )
        await redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️ Redis not available: {e}")
        redis_client = None
    
    # Создаем директории
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(INDEXES_DIR, exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs(os.environ['PROMETHEUS_MULTIPROC_DIR'], exist_ok=True)
    
    # Инициализируем улучшенный FeatureStore
    feature_store = EnhancedAsyncFeatureStore(BACKEND_URL, redis_client, BATCH_SIZE)
    
    # Инициализируем кэш моделей
    model_cache = ModelCache(max_size=MAX_MODEL_CACHE_SIZE, ttl=MODEL_TTL)
    
    # Прогреваем кэши
    try:
        await feature_store.warm_up_caches()
        logger.info("✅ Feature store caches warmed up")
    except Exception as e:
        logger.warning(f"⚠️ Failed to warm up caches: {e}")
    
    # Загружаем существующие индексы
    try:
        index_files = glob.glob(os.path.join(INDEXES_DIR, "model_*.ann"))
        for index_file in index_files:
            model_id_match = re.search(r'model_(\d+).ann', os.path.basename(index_file))
            if model_id_match:
                model_id = int(model_id_match.group(1))
                annoy_index_manager.load_index(model_id)
        logger.info(f"✅ Loaded {len(annoy_index_manager.indexes)} Annoy indexes")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load Annoy indexes: {e}")
    
    # Запускаем фоновые задачи
    asyncio.create_task(background_metrics_collector())
    asyncio.create_task(background_cache_cleanup())
    asyncio.create_task(background_prefetch())
    
    uptime = datetime.now() - start_time
    logger.info(f"🚀 ML Service started in {uptime.total_seconds():.2f}s")
    logger.info(f"   Environment: {ENVIRONMENT}")
    logger.info(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"   Models dir: {MODELS_DIR}")
    logger.info(f"   Redis: {'enabled' if redis_client else 'disabled'}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ML Service...")
    
    # Ждем завершения активных запросов
    for i in range(30):
        if active_requests == 0:
            break
        logger.info(f"Waiting for {active_requests} active requests to complete... ({i+1}/30)")
        await asyncio.sleep(1)
    
    if feature_store:
        await feature_store.close()
    
    if redis_client:
        await redis_client.close()
    
    logger.info("👋 ML Service stopped")


async def background_metrics_collector():
    """Фоновый сбор метрик системы"""
    while True:
        try:
            if ACTIVE_REQUESTS:
                ACTIVE_REQUESTS.set(active_requests)
            
            if MEMORY_USAGE:
                process = psutil.Process()
                memory_info = process.memory_info()
                MEMORY_USAGE.set(memory_info.rss)
            
            if CPU_USAGE:
                cpu_percent = psutil.cpu_percent(interval=1)
                CPU_USAGE.set(cpu_percent)
            
            await asyncio.sleep(15)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Metrics collector error: {e}")
            await asyncio.sleep(60)


async def background_cache_cleanup():
    """Фоновая очистка устаревших кэшей"""
    while True:
        try:
            if feature_store:
                await feature_store.cleanup_expired()
            
            # Очистка старых временных файлов
            try:
                temp_files = glob.glob(os.path.join("cache", "*.tmp"))
                for temp_file in temp_files:
                    file_age = datetime.now().timestamp() - os.path.getmtime(temp_file)
                    if file_age > 86400:  # старше 24 часов
                        os.remove(temp_file)
                        logger.debug(f"Removed old temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Temp file cleanup error: {e}")
            
            await asyncio.sleep(3600)  # каждый час
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            await asyncio.sleep(3600)


async def background_prefetch():
    """Фоновая предзагрузка популярных данных"""
    while True:
        try:
            if feature_store:
                await feature_store.prefetch_popular_items()
                logger.debug("Background prefetch completed")
            await asyncio.sleep(3600)  # каждый час
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Prefetch error: {e}")
            await asyncio.sleep(60)


# ============================================
# СОЗДАЕМ ПРИЛОЖЕНИЕ FASTAPI
# ============================================

app = FastAPI(
    title="Loyalty ML Service",
    version="3.2.0",
    lifespan=lifespan,
    docs_url="/api/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/api/redoc" if ENVIRONMENT != "production" else None,
    openapi_url="/api/openapi.json" if ENVIRONMENT != "production" else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:4201",
        "http://localhost:5000",
        "http://localhost:3000",
        "http://192.168.1.100:4200",
        "http://192.168.1.100:5000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# MIDDLEWARE
# ============================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    global active_requests
    request_id = hashlib.md5(f"{datetime.now().timestamp()}{id(request)}".encode()).hexdigest()[:8]
    
    logger.debug(f"[{request_id}] ➡️ {request.method} {request.url.path}")
    
    async with request_lock:
        active_requests += 1
    
    start_time = datetime.now()
    
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.debug(f"[{request_id}] ⬅️ {response.status_code} in {duration:.3f}s")
        return response
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"[{request_id}] ❌ Error in {duration:.3f}s: {e}")
        raise
    finally:
        async with request_lock:
            active_requests -= 1


# ============================================
# ЭНДПОИНТЫ
# ============================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    return {
        "service": "Loyalty ML Service",
        "version": "3.2.0",
        "status": "active",
        "environment": ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    redis_status = "healthy"
    try:
        if redis_client:
            await redis_client.ping()
    except:
        redis_status = "unhealthy"
    
    gpu_available = torch.cuda.is_available()
    gpu_memory = torch.cuda.memory_allocated() if gpu_available else None
    
    # Считаем uptime
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
            uptime = str(timedelta(seconds=uptime_seconds))
    except:
        uptime = "unknown"
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        device="cuda" if gpu_available else "cpu",
        redis=redis_status,
        models_loaded=len(model_cache.cache) if model_cache else 0,
        gpu_available=gpu_available,
        gpu_memory=gpu_memory,
        active_requests=active_requests,
        indexes_loaded=len(annoy_index_manager.indexes),
        memory_usage=memory_info.rss,
        cpu_usage=psutil.cpu_percent(),
        uptime=uptime
    )


@app.get("/metrics")
async def get_metrics():
    return JSONResponse(content=generate_latest(REGISTRY).decode())


@app.get("/ready")
async def readiness_check():
    """Проверка готовности к приему трафика"""
    if not feature_store:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "feature store not initialized"}
        )
    
    if not os.path.exists(MODELS_DIR):
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "models directory not found"}
        )
    
    return {"status": "ready", "timestamp": datetime.now().isoformat()}


@app.get("/live")
async def liveness_check():
    """Проверка живости сервиса"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.post("/train", response_model=Dict[str, Any])
async def train_model(data: TrainingConfig, background_tasks: BackgroundTasks):
    logger.info("="*60)
    logger.info(f"Training session {data.session_id}")
    logger.info("="*60)
    
    try:
        if TRAINING_COUNTER:
            TRAINING_COUNTER.labels(status='started').inc()
        
        # Подготовка данных
        interactions_data = []
        for i in data.dataset.interactions:
            if not i.ProductId:
                continue
            interactions_data.append({
                'customer_id': i.CustomerId,
                'product_id': i.ProductId,
                'timestamp': i.Timestamp,
                'quantity': i.Quantity,
                'price': i.Price
            })
        
        if not interactions_data:
            raise ValueError("No valid interactions")
        
        interactions_df = pd.DataFrame(interactions_data)
        
        # Фичи пользователей
        customers_df = pd.DataFrame([
            {'customer_id': c.PhoneNumber, **c.Features} 
            for c in data.dataset.customers if c.PhoneNumber
        ]).set_index('customer_id')
        
        # Фичи товаров
        products_df = pd.DataFrame([
            {'product_id': str(p.ProductId), **p.Features} 
            for p in data.dataset.products if p.ProductId
        ]).set_index('product_id')
        
        # Нормализация
        user_scaler = StandardScaler()
        item_scaler = StandardScaler()
        
        customers_df_scaled = pd.DataFrame(
            user_scaler.fit_transform(customers_df),
            index=customers_df.index,
            columns=customers_df.columns
        )
        
        products_df_scaled = pd.DataFrame(
            item_scaler.fit_transform(products_df),
            index=products_df.index,
            columns=products_df.columns
        )
        
        # Кодирование
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        all_users = interactions_df['customer_id'].unique()
        all_items = interactions_df['product_id'].unique()
        
        # Обработка отсутствующих товаров через MissingItemStrategy
        logger.info(f"Products in features before processing: {len(products_df_scaled)}")
        
        all_product_ids_from_interactions = set(str(pid) for pid in all_items)
        product_ids_in_features = set(products_df_scaled.index.astype(str))
        
        missing_products = all_product_ids_from_interactions - product_ids_in_features
        
        if missing_products:
            logger.warning(f"Found {len(missing_products)} products missing from product features")
            
            # Получаем умные дефолты
            smart_defaults = MissingItemStrategy.get_smart_defaults(products_df_scaled.columns.tolist())
            
            # Пытаемся получить реальные фичи для отсутствующих товаров
            if feature_store:
                for product_id in list(missing_products)[:100]:  # Ограничиваем для производительности
                    try:
                        real_features = await feature_store.get_item_features(product_id)
                        if real_features:
                            # Обновляем дефолты реальными значениями
                            for key, value in real_features.items():
                                if key in smart_defaults:
                                    smart_defaults[key] = value
                    except Exception as e:
                        logger.debug(f"Could not get features for {product_id}: {e}")
            
            dummy_data = []
            for product_id in missing_products:
                row = smart_defaults.copy()
                row['index'] = str(product_id)
                dummy_data.append(row)
            
            if dummy_data:
                dummy_df = pd.DataFrame(dummy_data)
                dummy_df = dummy_df.set_index('index')
                products_df_scaled = pd.concat([products_df_scaled, dummy_df])
                logger.info(f"Added {len(dummy_data)} products with smart defaults")
        
        user_encoder.fit(all_users)
        item_encoder.fit(all_items)
        
        # Разделение данных
        train_df, val_df = train_test_split(
            interactions_df, test_size=0.2, random_state=42
        )
        
        # Датасеты
        train_dataset = RecommendationDataset(
            train_df, user_encoder, item_encoder,
            customers_df_scaled, products_df_scaled, augment=True
        )
        
        val_dataset = RecommendationDataset(
            val_df, user_encoder, item_encoder,
            customers_df_scaled, products_df_scaled, augment=False
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples, Valid: {len(val_dataset)} samples")
        
        train_loader = DataLoader(
            train_dataset, batch_size=data.config.get('batch_size', 256),
            shuffle=True, num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=data.config.get('batch_size', 256),
            shuffle=False, num_workers=0
        )
        
        # Модель
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = ImprovedTwoTowerRecommender(
            num_users=len(user_encoder.classes_),
            num_items=len(item_encoder.classes_),
            user_feature_dim=len(customers_df.columns),
            item_feature_dim=len(products_df.columns),
            embedding_dim=data.config.get('embedding_size', 64),
            hidden_dim=data.config.get('hidden_size', 256),
            dropout=data.config.get('dropout', 0.3)
        ).to(device)
        
        # Обучение
        training_result = await train_model_with_validation(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=data.config,
            device=device,
            session_id=data.session_id
        )
        
        # Получаем эмбеддинги всех товаров
        model.eval()
        item_embeddings = []
        item_ids_list = []
        
        with torch.no_grad():
            for item_id in all_items:
                try:
                    item_idx = item_encoder.transform([item_id])[0]
                    item_id_str = str(item_id)
                    
                    if item_id_str in products_df_scaled.index:
                        item_feats = products_df_scaled.loc[item_id_str].values.astype(np.float32)
                    else:
                        smart_defaults = MissingItemStrategy.get_smart_defaults(products_df_scaled.columns.tolist())
                        item_feats = np.array([smart_defaults.get(col, 0.0) for col in products_df_scaled.columns], dtype=np.float32)
                    
                    item_tensor = torch.tensor([item_idx], dtype=torch.long).to(device)
                    feat_tensor = torch.tensor([item_feats], dtype=torch.float32).to(device)
                    
                    item_vector = model.get_item_embedding(item_tensor, feat_tensor).cpu().numpy()[0]
                    item_embeddings.append(item_vector)
                    item_ids_list.append(item_id)
                except Exception as e:
                    logger.warning(f"Error getting embedding for {item_id}: {e}")
                    continue
        
        # Строим Annoy индекс
        try:
            annoy_index_manager.build_index(
                data.session_id,
                np.array(item_embeddings),
                item_ids_list
            )
            logger.info(f"Built Annoy index for model {data.session_id}")
        except Exception as e:
            logger.error(f"Failed to build Annoy index: {e}")
        
        # Сохранение
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODELS_DIR, f"model_{data.session_id}_{timestamp}.pt")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'user_encoder': user_encoder,
            'item_encoder': item_encoder,
            'user_scaler': user_scaler,
            'item_scaler': item_scaler,
            'user_feature_columns': customers_df.columns.tolist(),
            'item_feature_columns': products_df.columns.tolist(),
            'config': data.config,
            'metrics': training_result.get('metrics', {}),
            'timestamp': timestamp,
            'version': '3.2.0',
            'embedding_dim': model.embedding_dim,
            'num_users': len(user_encoder.classes_),
            'num_items': len(item_encoder.classes_)
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training metrics: {training_result.get('metrics', {})}")
        
        # Уведомление бэкенду
        try:
            async with aiohttp.ClientSession() as session:
                notify_data = {
                    "sessionId": data.session_id,
                    "status": "completed",
                    "metrics": training_result.get('metrics', {}),
                    "modelPath": model_path,
                    "embeddingDim": model.embedding_dim
                }
                
                async with session.post(
                    f"{BACKEND_URL}/api/training/notify-complete",
                    json=notify_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Notified backend for session {data.session_id}")
                    else:
                        logger.error(f"Backend returned {response.status}")
        except Exception as e:
            logger.error(f"Failed to notify backend: {e}")
        
        if TRAINING_COUNTER:
            TRAINING_COUNTER.labels(status='completed').inc()
        
        return {
            "model_path": model_path, 
            "config": data.config,
            "metrics": training_result.get('metrics', {})
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        
        # Уведомление об ошибке
        try:
            async with aiohttp.ClientSession() as session:
                error_data = {
                    "sessionId": data.session_id,
                    "status": "failed",
                    "error": str(e)
                }
                
                await session.post(
                    f"{BACKEND_URL}/api/training/notify-complete",
                    json=error_data,
                    timeout=aiohttp.ClientTimeout(total=5)
                )
        except Exception as notify_error:
            logger.error(f"Failed to notify backend: {notify_error}")
        
        if TRAINING_COUNTER:
            TRAINING_COUNTER.labels(status='failed').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/{session_id}/cancel")
async def cancel_training(session_id: int):
    """Отмена тренировки"""
    try:
        logger.info(f"Cancelling training session {session_id}")
        # Здесь можно добавить логику отмены, если тренировка запущена в фоне
        return {"success": True, "message": f"Training {session_id} cancelled"}
    except Exception as e:
        logger.error(f"Error cancelling training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== ИСПРАВЛЕНИЕ: Обновлен эндпоинт ==========
@app.post("/incremental-train", response_model=IncrementalTrainingResponse)
async def incremental_train_endpoint(training_id: int, request: IncrementalTrainingRequest):
    """
    Инкрементальное дообучение модели на новых данных
    """
    try:
        logger.info("="*60)
        logger.info(f"Incremental training request received")
        logger.info(f"Training ID: {training_id}")
        logger.info(f"Base model ID: {request.base_model_id}")
        logger.info(f"New model ID: {request.new_model_id}")  # ← Логируем
        logger.info(f"New interactions count: {len(request.new_interactions)}")
        logger.info("="*60)
        
        # Проверяем, что new_model_id передан
        if not request.new_model_id:
            return IncrementalTrainingResponse(
                success=False,
                metrics={},
                model_path="",
                error="new_model_id is required"
            )
        
        result = await incremental_train(
            training_id=training_id,
            base_model_id=request.base_model_id,
            new_model_id=request.new_model_id,  # ← Передаем
            new_interactions=request.new_interactions,
            training_params=request.training_params
        )
        
        return IncrementalTrainingResponse(
            success=result['success'],
            metrics=result['metrics'],
            model_path=result['model_path'],
            error=result['error']
        )
            
    except Exception as e:
        logger.error(f"Incremental training endpoint error: {e}", exc_info=True)
        return IncrementalTrainingResponse(
            success=False,
            metrics={},
            model_path="",
            error=str(e)
        )
# ====================================================


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = datetime.now()
    
    # Лимит запросов
    if active_requests > 50:
        if PREDICTION_FALLBACK:
            PREDICTION_FALLBACK.labels(
                model_id=str(request.model_id), 
                reason='busy'
            ).inc()
        return JSONResponse(
            status_code=503, 
            content={"error": "Service busy", "active_requests": active_requests}
        )
    
    logger.info(f"=== PREDICTION REQUEST for {request.phone_number} ===")
    logger.info(f"  Model ID: {request.model_id}")
    logger.info(f"  Features: {request.features}")
    logger.info(f"  Top K: {request.top_k}")
    
    try:
        # Проверка кэша
        cache_key = f"pred_{request.phone_number}_{request.model_id}_{request.top_k}"
        if redis_client:
            try:
                cached = await redis_client.get(cache_key)
                if cached:
                    if PREDICTION_COUNTER:
                        PREDICTION_COUNTER.labels(
                            model_id=str(request.model_id), 
                            status='cached'
                        ).inc()
                    if CACHE_HITS:
                        CACHE_HITS.labels(cache_type='prediction').inc()
                    result = json.loads(cached)
                    result['cached'] = True
                    
                    logger.info(f"=== CACHED PREDICTION for {request.phone_number} ===")
                    logger.info(f"  Product IDs: {result['product_ids']}")
                    
                    return PredictionResponse(**result)
            except Exception as e:
                logger.warning(f"Redis error: {e}")
        
        if CACHE_MISSES:
            CACHE_MISSES.labels(cache_type='prediction').inc()
        
        # Загрузка модели
        model_files = glob.glob(os.path.join(MODELS_DIR, f"model_{request.model_id}_*.pt"))
        if not model_files:
            model_files = glob.glob(os.path.join(MODELS_DIR, "model_*.pt"))
            if not model_files:
                if PREDICTION_FALLBACK:
                    PREDICTION_FALLBACK.labels(
                        model_id=str(request.model_id), 
                        reason='no_model'
                    ).inc()
                
                fallback_result = {
                    "product_ids": ["FALLBACK-001", "FALLBACK-002", "FALLBACK-003"],
                    "top_product": "FALLBACK-001",
                    "discount": 10,
                    "probability": 0.3,
                    "score": 0.3,
                    "all_predictions": [],
                    "explanation": None,
                    "model_version": "fallback",
                    "cached": False,
                    "fallback": True
                }
                
                logger.info(f"=== FALLBACK PREDICTION for {request.phone_number} ===")
                
                return PredictionResponse(**fallback_result)
        
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        
        # Пытаемся получить из кэша
        if model_cache:
            cached_model = await model_cache.get(request.model_id)
            if cached_model:
                checkpoint = cached_model['checkpoint']
                model = cached_model['model']
                user_encoder = cached_model['user_encoder']
                item_encoder = cached_model['item_encoder']
                user_scaler = cached_model['user_scaler']
                item_scaler = cached_model['item_scaler']
                user_feature_columns = cached_model['user_feature_columns']
                item_feature_columns = cached_model['item_feature_columns']
                device = cached_model['device']
                logger.info(f"Loaded model {request.model_id} from cache")
            else:
                checkpoint = torch.load(latest_model, map_location='cpu')
                user_encoder = checkpoint['user_encoder']
                item_encoder = checkpoint['item_encoder']
                user_scaler = checkpoint.get('user_scaler')
                item_scaler = checkpoint.get('item_scaler')
                user_feature_columns = checkpoint['user_feature_columns']
                item_feature_columns = checkpoint['item_feature_columns']
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ImprovedTwoTowerRecommender(
                    num_users=len(user_encoder.classes_),
                    num_items=len(item_encoder.classes_),
                    user_feature_dim=len(user_feature_columns),
                    item_feature_dim=len(item_feature_columns),
                    embedding_dim=checkpoint['config'].get('embedding_size', 128),
                    hidden_dim=checkpoint['config'].get('hidden_size', 256)
                ).to(device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                await model_cache.set(request.model_id, {
                    'checkpoint': checkpoint,
                    'model': model,
                    'user_encoder': user_encoder,
                    'item_encoder': item_encoder,
                    'user_scaler': user_scaler,
                    'item_scaler': item_scaler,
                    'user_feature_columns': user_feature_columns,
                    'item_feature_columns': item_feature_columns,
                    'device': device
                })
                
                if feature_store:
                    feature_store.set_feature_columns(item_feature_columns)
                
                logger.info(f"Loaded model {request.model_id} from disk")
        else:
            # Fallback если кэш не инициализирован
            checkpoint = torch.load(latest_model, map_location='cpu')
            user_encoder = checkpoint['user_encoder']
            item_encoder = checkpoint['item_encoder']
            user_scaler = checkpoint.get('user_scaler')
            item_scaler = checkpoint.get('item_scaler')
            user_feature_columns = checkpoint['user_feature_columns']
            item_feature_columns = checkpoint['item_feature_columns']
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = ImprovedTwoTowerRecommender(
                num_users=len(user_encoder.classes_),
                num_items=len(item_encoder.classes_),
                user_feature_dim=len(user_feature_columns),
                item_feature_dim=len(item_feature_columns),
                embedding_dim=checkpoint['config'].get('embedding_size', 128),
                hidden_dim=checkpoint['config'].get('hidden_size', 256)
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        
        # Подготовка фичей пользователя
        user_features_df = pd.DataFrame([request.features])
        for col in user_feature_columns:
            if col not in user_features_df.columns:
                user_features_df[col] = 0.0
        
        user_features_df = user_features_df[user_feature_columns]
        
        if user_scaler:
            user_features_scaled = user_scaler.transform(user_features_df)
        else:
            user_features_scaled = user_features_df.values
        
        user_features_tensor = torch.tensor(user_features_scaled, dtype=torch.float32).to(device)
        
        # User ID
        try:
            user_id = user_encoder.transform([request.phone_number])[0]
            user_id_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
        except:
            logger.info(f"Cold start for {request.phone_number}")
            user_id_tensor = torch.tensor([0], dtype=torch.long).to(device)
            user_features_tensor = torch.zeros((1, len(user_feature_columns)), dtype=torch.float32).to(device)
        
        # Получаем эмбеддинг пользователя
        with torch.no_grad():
            user_embedding = model.get_user_embedding(
                user_id_tensor, 
                user_features_tensor
            ).cpu().numpy()[0]
        
        # Ищем ближайшие товары через Annoy
        nearest_items = annoy_index_manager.get_nearest(
            request.model_id, 
            user_embedding, 
            request.top_k * 2
        )
        
        if not nearest_items:
            # Fallback на прямой перебор
            logger.warning(f"Annoy index not found for model {request.model_id}, using brute force")
            
            all_items = item_encoder.classes_
            item_features_dict = {}
            
            if feature_store:
                item_features_dict = await feature_store.get_item_features_batch(all_items[:1000])
            
            predictions = []
            with torch.no_grad():
                for item_id in all_items[:1000]:
                    try:
                        item_idx = item_encoder.transform([item_id])[0]
                    except:
                        continue
                    
                    if item_id in item_features_dict:
                        feats = item_features_dict[item_id]
                        item_feats = np.array([feats.get(col, 0.5) for col in item_feature_columns], dtype=np.float32)
                    else:
                        if feature_store:
                            feats_dict = await feature_store.get_cold_start_features(item_id)
                            item_feats = np.array([feats_dict.get(col, 0.5) for col in item_feature_columns], dtype=np.float32)
                        else:
                            item_feats = np.zeros(len(item_feature_columns), dtype=np.float32)
                    
                    item_feats_tensor = torch.tensor([item_feats], dtype=torch.float32).to(device)
                    
                    if item_scaler:
                        item_feats_np = item_feats_tensor.cpu().numpy()
                        item_feats_scaled = item_scaler.transform(item_feats_np)
                        item_feats_tensor = torch.tensor(item_feats_scaled, dtype=torch.float32).to(device)
                    
                    score = model(
                        user_id_tensor,
                        torch.tensor([item_idx], dtype=torch.long).to(device),
                        user_features_tensor,
                        item_feats_tensor
                    ).item()
                    
                    predictions.append({
                        "product_id": item_id,
                        "score": float(score),
                        "probability": float(score)
                    })
            
            predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)[:request.top_k]
        else:
            predictions = nearest_items[:request.top_k]
        
        if not predictions:
            if PREDICTION_FALLBACK:
                PREDICTION_FALLBACK.labels(
                    model_id=str(request.model_id), 
                    reason='no_predictions'
                ).inc()
            raise HTTPException(status_code=500, detail="No predictions")
        
        # Расчет скидки
        discount = 10
        if predictions:
            discount = min(5 + int(predictions[0]['score'] * 30), 50)
        
        result = {
            "product_ids": [p['product_id'] for p in predictions],
            "top_product": predictions[0]['product_id'] if predictions else None,
            "discount": discount,
            "probability": predictions[0]['score'] if predictions else 0.5,
            "score": predictions[0]['score'] if predictions else 0.5,
            "all_predictions": predictions,
            "explanation": None,
            "model_version": checkpoint.get('version', '1.0.0'),
            "cached": False,
            "fallback": False
        }
        
        logger.info(f"=== PREDICTION RESULT for {request.phone_number} ===")
        logger.info(f"  Product IDs: {result['product_ids']}")
        logger.info(f"  Top product: {result['top_product']}")
        logger.info(f"  Discount: {result['discount']}%")
        
        # Сохраняем в кэш
        if redis_client:
            try:
                await redis_client.setex(
                    cache_key, 
                    PREDICTION_TTL, 
                    json.dumps(result, default=str)
                )
            except Exception as e:
                logger.warning(f"Cache error: {e}")
        
        # Метрики
        latency = (datetime.now() - start_time).total_seconds()
        if PREDICTION_LATENCY:
            PREDICTION_LATENCY.labels(model_id=str(request.model_id)).observe(latency)
        if PREDICTION_COUNTER:
            PREDICTION_COUNTER.labels(
                model_id=str(request.model_id), 
                status='success'
            ).inc()
        
        logger.info(f"Prediction completed in {latency:.2f}s")
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        if PREDICTION_COUNTER:
            PREDICTION_COUNTER.labels(
                model_id=str(request.model_id), 
                status='failed'
            ).inc()
        if PREDICTION_FALLBACK:
            PREDICTION_FALLBACK.labels(
                model_id=str(request.model_id), 
                reason='exception'
            ).inc()
        
        fallback_result = {
            "product_ids": ["FALLBACK-001", "FALLBACK-002", "FALLBACK-003"],
            "top_product": "FALLBACK-001",
            "discount": 10,
            "probability": 0.3,
            "score": 0.3,
            "all_predictions": [],
            "explanation": None,
            "model_version": "fallback",
            "cached": False,
            "fallback": True
        }
        
        logger.info(f"=== FALLBACK PREDICTION for {request.phone_number} ===")
        
        return PredictionResponse(**fallback_result)


@app.get("/models/{model_id}/info", response_model=ModelInfo)
async def get_model_info(model_id: int):
    """Детальная информация о модели"""
    try:
        # Ищем обычные модели
        model_files = glob.glob(os.path.join(MODELS_DIR, f"model_{model_id}_*.pt"))
        if not model_files:
            # Ищем инкрементальные модели
            model_files = glob.glob(os.path.join(MODELS_DIR, f"model_inc_*_{model_id}_*.pt"))
        
        if not model_files:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_files.sort(reverse=True)
        checkpoint = torch.load(model_files[0], map_location='cpu')
        
        metrics = checkpoint.get('metrics', {})
        
        return ModelInfo(
            model_id=model_id,
            version=checkpoint.get('version', '1.0.0'),
            created_at=datetime.strptime(checkpoint['timestamp'], '%Y%m%d_%H%M%S'),
            metrics=metrics,
            feature_importance=None,
            config=checkpoint['config']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """Очистка всех кэшей"""
    try:
        if model_cache:
            model_cache.cache.clear()
            model_cache.access_times.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if redis_client:
            await redis_client.flushdb()
        
        if feature_store:
            feature_store.cache.clear()
            feature_store.category_cache.clear()
            feature_store.popular_cache.clear()
        
        return {"status": "cache cleared", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-missing-features")
async def update_missing_features(request: Dict[str, Any]):
    """Обновление фичей для отсутствующих товаров"""
    try:
        model_id = request.get('model_id')
        feature_columns = request.get('feature_columns', [])
        
        if not model_id or not feature_columns:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required fields"}
            )
        
        logger.info(f"Update missing features request for model {model_id}")
        
        return {
            "success": True,
            "updated_count": 0,
            "message": "Update request received"
        }
        
    except Exception as e:
        logger.error(f"Error updating missing features: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/debug/cache-stats")
async def get_cache_stats():
    """Статистика кэша (только для отладки)"""
    if ENVIRONMENT == "production":
        raise HTTPException(status_code=403, detail="Not available in production")
    
    stats = {
        "model_cache_size": len(model_cache.cache) if model_cache else 0,
        "model_cache_keys": list(model_cache.cache.keys()) if model_cache else [],
        "feature_store_cache_size": len(feature_store.cache) if feature_store else 0,
        "annoy_indexes": list(annoy_index_manager.indexes.keys()),
        "active_requests": active_requests,
        "redis_connected": redis_client is not None
    }
    
    return stats


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    workers = min(MAX_WORKERS, multiprocessing.cpu_count())
    
    logger.info(f"Starting ML Service with {workers} workers")
    logger.info(f"Environment: {ENVIRONMENT}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=ENVIRONMENT != "production",
        workers=1,  # Для Windows лучше 1, иначе проблемы с multiprocessing
        log_level="info" if ENVIRONMENT == "production" else "debug",
        limit_max_requests=10000,
        timeout_keep_alive=30
    )
