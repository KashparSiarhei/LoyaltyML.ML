import numpy as np
from annoy import AnnoyIndex
import torch
import logging
from typing import List, Dict, Any, Optional
import os
import pickle

logger = logging.getLogger(__name__)

class AnnoyIndexManager:
    """Управление индексами Annoy для быстрого поиска"""
    
    def __init__(self, embedding_dim: int = 64, metric: str = 'angular'):
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.indexes: Dict[int, AnnoyIndex] = {}
        self.item_mappings: Dict[int, Dict[int, str]] = {}  # model_id -> {index: item_id}
        
    def build_index(self, model_id: int, item_embeddings: np.ndarray, item_ids: List[str]):
        """Построение индекса для модели"""
        try:
            index = AnnoyIndex(self.embedding_dim, self.metric)
            
            for i, emb in enumerate(item_embeddings):
                index.add_item(i, emb)
            
            index.build(100)  # 100 деревьев для точности
            
            self.indexes[model_id] = index
            self.item_mappings[model_id] = {i: item_id for i, item_id in enumerate(item_ids)}
            
            # Сохраняем индекс на диск
            os.makedirs("indexes", exist_ok=True)
            index.save(f"indexes/model_{model_id}.ann")
            
            with open(f"indexes/model_{model_id}_mapping.pkl", 'wb') as f:
                pickle.dump(self.item_mappings[model_id], f)
            
            logger.info(f"Built Annoy index for model {model_id} with {len(item_ids)} items")
            return True
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def load_index(self, model_id: int) -> bool:
        """Загрузка индекса с диска"""
        try:
            index_path = f"indexes/model_{model_id}.ann"
            mapping_path = f"indexes/model_{model_id}_mapping.pkl"
            
            if not os.path.exists(index_path) or not os.path.exists(mapping_path):
                return False
            
            index = AnnoyIndex(self.embedding_dim, self.metric)
            index.load(index_path)
            
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f)
            
            self.indexes[model_id] = index
            self.item_mappings[model_id] = mapping
            
            logger.info(f"Loaded Annoy index for model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_nearest(self, model_id: int, user_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, float]]:
        """Поиск ближайших товаров"""
        if model_id not in self.indexes:
            if not self.load_index(model_id):
                return []
        
        index = self.indexes[model_id]
        mapping = self.item_mappings[model_id]
        
        indices, distances = index.get_nns_by_vector(
            user_embedding, 
            top_k, 
            include_distances=True
        )
        
        # Конвертируем расстояние в сходство
        results = []
        for idx, dist in zip(indices, distances):
            item_id = mapping[idx]
            # angular distance to cosine similarity
            similarity = 1.0 - (dist * dist) / 2.0
            results.append({
                "product_id": item_id,
                "score": float(similarity),
                "distance": float(dist)
            })
        
        return results

# Глобальный экземпляр
annoy_index_manager = AnnoyIndexManager()