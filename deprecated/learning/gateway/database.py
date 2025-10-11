"""
Database models and connection handling for VoiceStand Learning System
Replaces in-memory storage with persistent PostgreSQL storage
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import asyncpg
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Database connection configuration
DATABASE_URL = os.getenv(
    "LEARNING_DB_URL",
    "postgresql://voicestand:learning_pass@localhost:5433/voicestand_learning"
)

class DatabaseConnection:
    """Manages PostgreSQL database connections with connection pooling"""

    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize the database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("✅ Database connection pool initialized")

            # Test the connection
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                logger.info(f"📊 Connected to: {result}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close the database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("🔒 Database connection pool closed")

    async def get_connection(self):
        """Get a database connection from the pool"""
        if not self.pool:
            await self.initialize()
        return self.pool.acquire()

# Global database instance
db = DatabaseConnection()

class RecognitionHistoryModel(BaseModel):
    """Model for recognition history records"""
    id: Optional[int] = None
    recognition_id: str
    audio_features: Optional[List[float]] = None
    recognized_text: str
    confidence: float
    model_used: str
    processing_time_ms: int
    speaker_id: Optional[str] = None
    is_uk_english: bool = False
    ground_truth: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class LearningPatternModel(BaseModel):
    """Model for learning patterns"""
    id: Optional[int] = None
    pattern_type: str
    source_text: str
    target_text: str
    confidence: float
    usage_count: int = 1
    accuracy_improvement: float = 0.0
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ModelPerformanceModel(BaseModel):
    """Model for model performance metrics"""
    id: Optional[int] = None
    model_name: str
    accuracy: float
    uk_accuracy: Optional[float] = None
    weight: float
    sample_count: int = 0
    total_processing_time_ms: int = 0
    last_updated: Optional[datetime] = None
    created_at: Optional[datetime] = None

class SystemMetricModel(BaseModel):
    """Model for system metrics"""
    id: Optional[int] = None
    metric_name: str
    metric_value: float
    metadata: Optional[Dict] = None
    timestamp: Optional[datetime] = None

class ActivityLogModel(BaseModel):
    """Model for activity log entries"""
    id: Optional[int] = None
    activity_type: str
    message: str
    metadata: Optional[Dict] = None
    timestamp: Optional[datetime] = None

class LearningInsightModel(BaseModel):
    """Model for learning insights"""
    id: Optional[int] = None
    insight_type: str
    description: str
    confidence: float
    recommendations: Dict
    requires_retraining: bool = False
    is_active: bool = True
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

class DatabaseOperations:
    """Database operations for the learning system"""

    @staticmethod
    async def store_recognition(recognition: RecognitionHistoryModel) -> str:
        """Store a recognition result in the database"""
        async with db.get_connection() as conn:
            # Convert audio features to vector format if provided
            features_vector = None
            if recognition.audio_features:
                features_vector = recognition.audio_features

            result = await conn.fetchrow("""
                INSERT INTO recognition_history
                (recognition_id, audio_features, recognized_text, confidence,
                 model_used, processing_time_ms, speaker_id, is_uk_english, ground_truth)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id, recognition_id
            """,
                recognition.recognition_id,
                features_vector,
                recognition.recognized_text,
                recognition.confidence,
                recognition.model_used,
                recognition.processing_time_ms,
                recognition.speaker_id,
                recognition.is_uk_english,
                recognition.ground_truth
            )

            logger.info(f"✅ Stored recognition {result['recognition_id']}")
            return result['recognition_id']

    @staticmethod
    async def get_model_performance() -> List[ModelPerformanceModel]:
        """Get all model performance metrics"""
        async with db.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT * FROM model_performance
                ORDER BY last_updated DESC
            """)

            return [ModelPerformanceModel(**dict(row)) for row in rows]

    @staticmethod
    async def update_model_performance(model_name: str, accuracy: float,
                                     uk_accuracy: Optional[float] = None,
                                     weight: Optional[float] = None,
                                     sample_count_delta: int = 0) -> None:
        """Update model performance metrics"""
        async with db.get_connection() as conn:
            await conn.execute("""
                UPDATE model_performance
                SET accuracy = COALESCE($2, accuracy),
                    uk_accuracy = COALESCE($3, uk_accuracy),
                    weight = COALESCE($4, weight),
                    sample_count = sample_count + $5,
                    last_updated = NOW()
                WHERE model_name = $1
            """, model_name, accuracy, uk_accuracy, weight, sample_count_delta)

            logger.info(f"✅ Updated performance for model {model_name}")

    @staticmethod
    async def get_system_metrics() -> Dict[str, float]:
        """Get current system metrics"""
        async with db.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT metric_name, metric_value
                FROM system_metrics
                WHERE timestamp = (
                    SELECT MAX(timestamp)
                    FROM system_metrics s2
                    WHERE s2.metric_name = system_metrics.metric_name
                )
            """)

            return {row['metric_name']: row['metric_value'] for row in rows}

    @staticmethod
    async def update_system_metric(metric_name: str, value: float,
                                 metadata: Optional[Dict] = None) -> None:
        """Update a system metric"""
        async with db.get_connection() as conn:
            await conn.execute("""
                INSERT INTO system_metrics (metric_name, metric_value, metadata)
                VALUES ($1, $2, $3)
            """, metric_name, value, json.dumps(metadata) if metadata else None)

            logger.debug(f"📊 Updated metric {metric_name}: {value}")

    @staticmethod
    async def add_learning_pattern(pattern: LearningPatternModel) -> int:
        """Add a new learning pattern"""
        async with db.get_connection() as conn:
            result = await conn.fetchrow("""
                INSERT INTO learning_patterns
                (pattern_type, source_text, target_text, confidence, usage_count, accuracy_improvement)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """,
                pattern.pattern_type,
                pattern.source_text,
                pattern.target_text,
                pattern.confidence,
                pattern.usage_count,
                pattern.accuracy_improvement
            )

            logger.info(f"✅ Added learning pattern: {pattern.pattern_type}")
            return result['id']

    @staticmethod
    async def get_learning_patterns(pattern_type: Optional[str] = None) -> List[LearningPatternModel]:
        """Get learning patterns, optionally filtered by type"""
        async with db.get_connection() as conn:
            if pattern_type:
                rows = await conn.fetch("""
                    SELECT * FROM learning_patterns
                    WHERE pattern_type = $1 AND is_active = true
                    ORDER BY usage_count DESC, created_at DESC
                """, pattern_type)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM learning_patterns
                    WHERE is_active = true
                    ORDER BY usage_count DESC, created_at DESC
                """)

            return [LearningPatternModel(**dict(row)) for row in rows]

    @staticmethod
    async def count_patterns_learned() -> int:
        """Count total learning patterns"""
        async with db.get_connection() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM learning_patterns WHERE is_active = true
            """)
            return result or 0

    @staticmethod
    async def add_activity(activity_type: str, message: str,
                          metadata: Optional[Dict] = None) -> None:
        """Add an activity log entry"""
        async with db.get_connection() as conn:
            await conn.execute("""
                INSERT INTO activity_log (activity_type, message, metadata)
                VALUES ($1, $2, $3)
            """, activity_type, message, json.dumps(metadata) if metadata else None)

            logger.debug(f"📝 Added activity: {activity_type}")

    @staticmethod
    async def get_recent_activity(limit: int = 20) -> List[ActivityLogModel]:
        """Get recent activity log entries"""
        async with db.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT * FROM activity_log
                ORDER BY timestamp DESC
                LIMIT $1
            """, limit)

            activities = []
            for row in rows:
                activity_dict = dict(row)
                if activity_dict['metadata']:
                    activity_dict['metadata'] = json.loads(activity_dict['metadata'])
                activities.append(ActivityLogModel(**activity_dict))

            return activities

    @staticmethod
    async def get_learning_insights() -> List[LearningInsightModel]:
        """Get current learning insights"""
        async with db.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT * FROM learning_insights
                WHERE is_active = true
                AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY confidence DESC, created_at DESC
            """)

            insights = []
            for row in rows:
                insight_dict = dict(row)
                if insight_dict['recommendations']:
                    insight_dict['recommendations'] = json.loads(insight_dict['recommendations'])
                insights.append(LearningInsightModel(**insight_dict))

            return insights

    @staticmethod
    async def add_learning_insight(insight: LearningInsightModel) -> int:
        """Add a new learning insight"""
        async with db.get_connection() as conn:
            result = await conn.fetchrow("""
                INSERT INTO learning_insights
                (insight_type, description, confidence, recommendations, requires_retraining, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """,
                insight.insight_type,
                insight.description,
                insight.confidence,
                json.dumps(insight.recommendations),
                insight.requires_retraining,
                insight.expires_at
            )

            logger.info(f"✅ Added insight: {insight.insight_type}")
            return result['id']

    @staticmethod
    async def find_similar_audio_features(features: List[float], limit: int = 5) -> List[Dict]:
        """Find similar audio features using vector similarity"""
        async with db.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT recognition_id, recognized_text, confidence, model_used,
                       audio_features <-> $1 AS distance
                FROM recognition_history
                WHERE audio_features IS NOT NULL
                ORDER BY audio_features <-> $1
                LIMIT $2
            """, features, limit)

            return [dict(row) for row in rows]

    @staticmethod
    async def get_performance_history(model_name: Optional[str] = None,
                                    days: int = 30) -> List[Dict]:
        """Get historical performance data"""
        async with db.get_connection() as conn:
            if model_name:
                rows = await conn.fetch("""
                    SELECT metric_name, metric_value, timestamp, metadata
                    FROM system_metrics
                    WHERE timestamp >= NOW() - INTERVAL '%s days'
                    AND metadata->>'model' = $2
                    ORDER BY timestamp ASC
                """, days, model_name)
            else:
                rows = await conn.fetch("""
                    SELECT metric_name, metric_value, timestamp, metadata
                    FROM system_metrics
                    WHERE timestamp >= NOW() - INTERVAL '%s days'
                    ORDER BY timestamp ASC
                """, days)

            results = []
            for row in rows:
                result_dict = dict(row)
                if result_dict['metadata']:
                    result_dict['metadata'] = json.loads(result_dict['metadata'])
                results.append(result_dict)

            return results

# Async context manager for database lifecycle
class DatabaseManager:
    """Context manager for database operations"""

    async def __aenter__(self):
        await db.initialize()
        return db

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await db.close()

# Initialize database on module import
async def init_database():
    """Initialize database connection"""
    await db.initialize()

# Cleanup database on shutdown
async def cleanup_database():
    """Cleanup database connections"""
    await db.close()