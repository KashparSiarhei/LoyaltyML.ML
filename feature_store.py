import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
            col_lower = col.lower()
            if 'price' in col_lower:
                defaults[col] = 500.0  # средняя цена
            elif 'popularity' in col_lower:
                defaults[col] = 0.3    # ниже среднего для новых товаров
            elif 'margin' in col_lower:
                defaults[col] = 25.0    # средняя маржа
            elif 'category' in col_lower or 'cat' in col_lower:
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


class AsyncFeatureStore:
    """Асинхронное получение фичей с кэшированием и предзагрузкой"""
    
    def __init__(self, backend_url: str, redis_client=None, batch_size: int = 100):
        self.backend_url = backend_url
        self.redis = redis_client
        self.batch_size = batch_size
        self.cache: Dict[str, Dict] = {}
        self.feature_columns = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._popular_items_cache: Dict[str, List] = {}
        self._last_update = datetime.min
        
    async def get_session(self) -> aiohttp.ClientSession:
        """Получение или создание сессии"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Закрытие сессии"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def prefetch_popular_items(self, template_id: int = 2, force: bool = False):
        """Предзагрузка популярных товаров (запускать в фоне)"""
        now = datetime.now()
        if not force and (now - self._last_update) < timedelta(hours=1):
            return
        
        try:
            session = await self.get_session()
            async with session.post(
                f"{self.backend_url}/api/querytemplate/{template_id}/execute",
                json={"limit": 1000},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    rows = data.get('rows', [])
                    
                    # Сохраняем в кэш
                    tasks = []
                    for row in rows:
                        item_id = row.get('product_id') or row.get('id')
                        if item_id:
                            features = self._extract_features(row)
                            self.cache[item_id] = features
                            if self.redis:
                                tasks.append(
                                    self.redis.setex(
                                        f"item_features:{item_id}",
                                        3600,
                                        json.dumps(features)
                                    )
                                )
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    self._popular_items_cache['items'] = [r.get('product_id') or r.get('id') for r in rows if r.get('product_id') or r.get('id')]
                    self._last_update = now
                    
                    logger.info(f"Prefetched {len(rows)} popular items")
        except Exception as e:
            logger.error(f"Failed to prefetch items: {e}")
    
    def _extract_features(self, row: Dict) -> Dict[str, float]:
        """Извлечение числовых фичей из строки"""
        features = {}
        for key, value in row.items():
            if key not in ['product_id', 'id', 'name', 'description', 'created_at']:
                try:
                    features[key] = float(value)
                except (TypeError, ValueError):
                    pass
        
        # Если нет числовых фичей, добавляем базовые
        if not features:
            features = {
                'price': float(row.get('price', 100)),
                'popularity': float(row.get('popularity', 0.5)),
                'margin': float(row.get('margin', 30))
            }
        
        return features
    
    async def get_item_features(self, item_id: str) -> Optional[Dict[str, float]]:
        """Получение фичей для одного товара"""
        # Проверяем кэш
        async with self._lock:
            if item_id in self.cache:
                return self.cache[item_id]
            
            if self.redis:
                try:
                    cached = await self.redis.get(f"item_features:{item_id}")
                    if cached:
                        features = json.loads(cached)
                        self.cache[item_id] = features
                        return features
                except Exception as e:
                    logger.warning(f"Redis error for {item_id}: {e}")
        
        # Получаем через API
        result = await self.get_item_features_batch([item_id])
        return result.get(item_id)
    
    async def get_item_features_batch(self, item_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Получение фичей для нескольких товаров параллельно"""
        if not item_ids:
            return {}
        
        result = {}
        missing_ids = []
        
        # Проверяем кэш
        async with self._lock:
            for item_id in item_ids:
                if item_id in self.cache:
                    result[item_id] = self.cache[item_id]
                elif self.redis:
                    try:
                        cached = await self.redis.get(f"item_features:{item_id}")
                        if cached:
                            features = json.loads(cached)
                            self.cache[item_id] = features
                            result[item_id] = features
                        else:
                            missing_ids.append(item_id)
                    except:
                        missing_ids.append(item_id)
                else:
                    missing_ids.append(item_id)
        
        # Получаем недостающие батчами
        if missing_ids:
            await self._fetch_batches(missing_ids, result)
        
        return result
    
    async def _fetch_batches(self, item_ids: List[str], result: Dict):
        """Асинхронное получение батчами"""
        session = await self.get_session()
        
        # Разбиваем на батчи
        batches = [item_ids[i:i + self.batch_size] 
                  for i in range(0, len(item_ids), self.batch_size)]
        
        tasks = []
        for batch in batches:
            tasks.append(self._fetch_batch(session, batch))
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Объединяем результаты
        async with self._lock:
            for batch_result in batch_results:
                if isinstance(batch_result, dict):
                    result.update(batch_result)
                    for item_id, features in batch_result.items():
                        self.cache[item_id] = features
    
    async def _fetch_batch(self, session: aiohttp.ClientSession, batch: List[str]) -> Dict:
        """Получение одного батча"""
        try:
            async with session.post(
                f"{self.backend_url}/api/querytemplate/1/execute",  # Шаблон для фичей
                json={"product_ids": batch},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    rows = data.get('rows', [])
                    
                    batch_result = {}
                    redis_tasks = []
                    
                    for row in rows:
                        item_id = row.get('product_id') or row.get('id')
                        if item_id:
                            features = self._extract_features(row)
                            batch_result[item_id] = features
                            
                            if self.redis:
                                redis_tasks.append(
                                    self.redis.setex(
                                        f"item_features:{item_id}",
                                        3600,
                                        json.dumps(features)
                                    )
                                )
                    
                    if redis_tasks:
                        await asyncio.gather(*redis_tasks, return_exceptions=True)
                    
                    return batch_result
                return {}
        except Exception as e:
            logger.error(f"Batch fetch error: {e}")
            return {}
    
    async def get_user_features(self, user_id: str) -> Optional[Dict[str, float]]:
        """Получение фичей пользователя"""
        try:
            session = await self.get_session()
            async with session.get(
                f"{self.backend_url}/api/customer/{user_id}/features",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('features', {})
                return None
        except Exception as e:
            logger.error(f"Error getting user features for {user_id}: {e}")
            return None
    
    async def get_cold_start_features(self, item_id: str) -> Dict[str, float]:
        """Cold start - возвращаем дефолтные фичи"""
        if self.feature_columns:
            # Используем умные дефолты на основе названий колонок
            defaults = self._get_smart_defaults(self.feature_columns)
            return defaults
        return {
            'price': 500.0,
            'popularity': 0.3,
            'margin': 25.0,
            'category_id': -1.0
        }
    
    def _get_smart_defaults(self, columns: List[str]) -> Dict[str, float]:
        """Умные дефолтные значения"""
        defaults = {}
        for col in columns:
            col_lower = col.lower()
            if 'price' in col_lower:
                defaults[col] = 500.0
            elif 'popularity' in col_lower:
                defaults[col] = 0.3
            elif 'margin' in col_lower:
                defaults[col] = 25.0
            elif 'category' in col_lower:
                defaults[col] = -1.0
            else:
                defaults[col] = 0.0
        return defaults
    
    def set_feature_columns(self, columns: List[str]):
        """Установка колонок фичей (из обученной модели)"""
        self.feature_columns = columns
    
    async def cleanup_expired(self):
        """Очистка устаревших записей в кэше"""
        # Реализация при необходимости
        pass


class EnhancedAsyncFeatureStore(AsyncFeatureStore):
    """Расширенный FeatureStore с дополнительными методами для поиска по категориям и популярным товарам"""
    
    def __init__(self, backend_url: str, redis_client=None, batch_size: int = 100):
        super().__init__(backend_url, redis_client, batch_size)
        self.category_cache: Dict[str, List[Dict]] = {}
        self.popular_cache: List[Dict] = []
        self._category_last_update = datetime.min
        self._popular_last_update = datetime.min
    
    async def get_items_by_category(self, category_id: str, limit: int = 50) -> List[Dict]:
        """Получить товары по категории"""
        now = datetime.now()
        cache_key = f"category_{category_id}"
        
        # Проверяем кэш
        if cache_key in self.category_cache:
            if (now - self._category_last_update) < timedelta(hours=1):
                return self.category_cache[cache_key][:limit]
        
        try:
            session = await self.get_session()
            # Используем шаблон 3 для товаров по категории
            async with session.post(
                f"{self.backend_url}/api/querytemplate/3/execute",
                json={"category_id": category_id, "limit": limit},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    rows = data.get('rows', [])
                    
                    items = []
                    for row in rows:
                        item_id = row.get('product_id') or row.get('id')
                        if item_id:
                            features = self._extract_features(row)
                            items.append({
                                'id': item_id,
                                'features': features
                            })
                    
                    self.category_cache[cache_key] = items
                    self._category_last_update = now
                    return items
        except Exception as e:
            logger.error(f"Error getting items by category {category_id}: {e}")
        
        return []
    
    async def get_popular_items(self, limit: int = 100) -> List[Dict]:
        """Получить популярные товары"""
        now = datetime.now()
        
        if self.popular_cache and (now - self._popular_last_update) < timedelta(hours=1):
            return self.popular_cache[:limit]
        
        try:
            session = await self.get_session()
            # Используем шаблон 2 для популярных товаров
            async with session.post(
                f"{self.backend_url}/api/querytemplate/2/execute",
                json={"limit": limit},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    rows = data.get('rows', [])
                    
                    items = []
                    for row in rows:
                        item_id = row.get('product_id') or row.get('id')
                        if item_id:
                            features = self._extract_features(row)
                            items.append({
                                'id': item_id,
                                'features': features
                            })
                    
                    self.popular_cache = items
                    self._popular_last_update = now
                    return items
        except Exception as e:
            logger.error(f"Error getting popular items: {e}")
        
        return []
    
    async def warm_up_caches(self):
        """Прогрев кэшей при старте"""
        logger.info("Warming up caches...")
        await self.get_popular_items()
        logger.info(f"Caches warmed up: {len(self.popular_cache)} popular items")