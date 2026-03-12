import logging
from typing import List, Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class MissingItemStrategy:
    """Стратегии обработки отсутствующих товаров"""
    
    @staticmethod
    async def get_similar_items_features(item_id: str, feature_store) -> Optional[Dict[str, float]]:
        """Получить фичи похожих товаров (по категории)"""
        try:
            # Пытаемся найти товары из той же категории
            similar_items = await feature_store.get_items_by_category(item_id)
            if similar_items:
                # Усредняем фичи похожих товаров
                avg_features = {}
                count = 0
                for item in similar_items:
                    if item_id != item['id']:  # исключаем сам товар
                        for key, value in item.get('features', {}).items():
                            if key not in avg_features:
                                avg_features[key] = 0
                            avg_features[key] += value
                        count += 1
                
                if count > 0:
                    for key in avg_features:
                        avg_features[key] /= count
                    logger.info(f"Generated averaged features for missing item {item_id} from {count} similar items")
                    return avg_features
        except Exception as e:
            logger.error(f"Error getting similar items for {item_id}: {e}")
        return None
    
    @staticmethod
    async def get_popular_items_stats(feature_store) -> Dict[str, float]:
        """Получить статистику по популярным товарам"""
        try:
            popular_items = await feature_store.get_popular_items(limit=100)
            if popular_items:
                # Вычисляем средние значения фичей по популярным товарам
                avg_features = {}
                count = 0
                for item in popular_items:
                    features = item.get('features', {})
                    for key, value in features.items():
                        if key not in avg_features:
                            avg_features[key] = 0
                        avg_features[key] += value
                    count += 1
                
                if count > 0:
                    for key in avg_features:
                        avg_features[key] /= count
                    logger.info(f"Generated average features from {count} popular items")
                    return avg_features
        except Exception as e:
            logger.error(f"Error getting popular items stats: {e}")
        return {}
    
    @staticmethod
    async def get_category_defaults(category_id: str, feature_store) -> Dict[str, float]:
        """Получить дефолтные фичи для категории"""
        try:
            # Пытаемся найти типичные значения для категории
            category_items = await feature_store.get_items_by_category(category_id, limit=50)
            if category_items:
                avg_features = {}
                count = 0
                for item in category_items:
                    features = item.get('features', {})
                    for key, value in features.items():
                        if key not in avg_features:
                            avg_features[key] = 0
                        avg_features[key] += value
                    count += 1
                
                if count > 0:
                    for key in avg_features:
                        avg_features[key] /= count
                    logger.info(f"Generated category defaults for {category_id} from {count} items")
                    return avg_features
        except Exception as e:
            logger.error(f"Error getting category defaults for {category_id}: {e}")
        return {}
    
    @staticmethod
    def get_smart_defaults(feature_columns: List[str]) -> Dict[str, float]:
        """Умные дефолтные значения на основе известных диапазонов"""
        defaults = {}
        for col in feature_columns:
            if 'price' in col.lower():
                defaults[col] = 500.0  # средняя цена
            elif 'popularity' in col.lower():
                defaults[col] = 0.3    # ниже среднего для новых товаров
            elif 'margin' in col.lower():
                defaults[col] = 25.0    # средняя маржа
            elif 'category' in col.lower():
                defaults[col] = -1.0     # неизвестная категория
            else:
                defaults[col] = 0.0
        return defaults
    
    @staticmethod
    def create_feature_vector_from_defaults(defaults: Dict[str, float], columns: List[str]) -> np.ndarray:
        """Создание вектора фичей из дефолтов в правильном порядке"""
        vector = np.zeros(len(columns), dtype=np.float32)
        for i, col in enumerate(columns):
            vector[i] = defaults.get(col, 0.0)
        return vector