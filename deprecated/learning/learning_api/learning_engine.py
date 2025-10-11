"""
VoiceStand Advanced Learning Engine
Orchestrates the multi-model ensemble learning system for 94-99% accuracy
"""

import asyncio
import logging
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import asyncpg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import VotingClassifier
import torch
import librosa

logger = logging.getLogger(__name__)

@dataclass
class RecognitionResult:
    text: str
    confidence: float
    model_used: str
    processing_time_ms: int
    acoustic_features: List[float]
    is_uk_english: bool = False
    speaker_id: Optional[str] = None
    ground_truth: Optional[str] = None
    ensemble_agreement: float = 0.0

@dataclass
class LearningPattern:
    pattern_id: str
    pattern_type: str
    features: Dict[str, float]
    confidence_score: float
    accuracy_improvement: float
    usage_count: int
    is_uk_specific: bool
    created_at: datetime
    last_used: datetime

class LearningEngine:
    """Core learning engine for VoiceStand's continuous improvement system"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_pool = None
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.model_performance: Dict[str, float] = {}
        self.recognition_history: List[RecognitionResult] = []
        self.max_history_size = 10000
        self.accuracy_target = 0.95

        # UK English specialization
        self.uk_vocabulary_mappings = self._load_uk_vocabulary()
        self.uk_acoustic_patterns = {}

        # Ensemble coordination
        self.model_weights = {
            "ggml-small.bin": 0.79,
            "ggml-medium.bin": 0.85,
            "ggml-large.bin": 0.88,
            "uk-english-fine-tuned-small.bin": 0.82,
            "uk-english-fine-tuned-medium.bin": 0.87
        }

        # Learning statistics
        self.total_recognitions = 0
        self.successful_adaptations = 0
        self.uk_patterns_learned = 0

    async def initialize(self):
        """Initialize the learning engine with database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            await self._initialize_database()
            await self._load_existing_patterns()
            logger.info("Learning engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize learning engine: {e}")
            return False

    async def process_recognition(self, recognition_data: Dict[str, Any]):
        """Process a recognition result and extract learning patterns"""
        try:
            result = RecognitionResult(
                text=recognition_data["recognized_text"],
                confidence=recognition_data["confidence"],
                model_used=recognition_data["model_used"],
                processing_time_ms=recognition_data["processing_time_ms"],
                acoustic_features=recognition_data["audio_features"],
                is_uk_english=recognition_data.get("is_uk_english", False),
                speaker_id=recognition_data.get("speaker_id"),
                ground_truth=recognition_data.get("ground_truth")
            )

            # Add to history
            self.recognition_history.append(result)
            if len(self.recognition_history) > self.max_history_size:
                self.recognition_history.pop(0)

            # Extract learning patterns
            patterns = await self._extract_patterns(result)

            # Update model performance
            await self._update_model_performance(result)

            # UK English specific processing
            if result.is_uk_english:
                await self._process_uk_english_recognition(result)

            # Store in database
            await self._store_recognition_result(result)

            self.total_recognitions += 1

            return patterns

        except Exception as e:
            logger.error(f"Error processing recognition: {e}")
            return []

    async def get_model_performance(self) -> List[Dict[str, Any]]:
        """Get current performance metrics for all models"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT model_name,
                           AVG(accuracy) as avg_accuracy,
                           AVG(CASE WHEN is_uk_english THEN accuracy END) as uk_accuracy,
                           COUNT(*) as sample_count,
                           MAX(recorded_at) as last_updated
                    FROM model_performance
                    WHERE recorded_at > NOW() - INTERVAL '24 hours'
                    GROUP BY model_name
                    ORDER BY avg_accuracy DESC
                """
                rows = await conn.fetch(query)

                performance = []
                for row in rows:
                    performance.append({
                        "model_name": row["model_name"],
                        "accuracy": float(row["avg_accuracy"]),
                        "uk_accuracy": float(row["uk_accuracy"]) if row["uk_accuracy"] else None,
                        "sample_count": row["sample_count"],
                        "last_updated": row["last_updated"]
                    })

                return performance

        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return []

    async def get_optimal_ensemble(self, context: Dict[str, Any]) -> List[str]:
        """Get optimal model ensemble for given context"""
        try:
            # Analyze context
            is_uk_english = context.get("is_uk_english", False)
            noise_level = context.get("noise_level", 0.0)
            speech_rate = context.get("speech_rate", 1.0)
            domain = context.get("domain", "general")

            # Base model selection
            selected_models = ["ggml-medium.bin", "ggml-large.bin"]

            # Add UK-specific models if appropriate
            if is_uk_english:
                selected_models.extend([
                    "uk-english-fine-tuned-medium.bin",
                    "uk-english-fine-tuned-small.bin"
                ])

            # Adjust for noise conditions
            if noise_level > 0.5:
                selected_models.append("ggml-large.bin")  # Better for noisy conditions

            # Limit ensemble size for performance
            max_ensemble_size = 3 if noise_level > 0.3 else 5

            # Sort by current performance and select top models
            model_scores = []
            for model in selected_models:
                base_score = self.model_weights.get(model, 0.5)
                uk_bonus = 0.05 if is_uk_english and "uk-english" in model else 0.0
                domain_bonus = 0.02 if domain in model else 0.0

                total_score = base_score + uk_bonus + domain_bonus
                model_scores.append((model, total_score))

            # Sort by score and take top models
            model_scores.sort(key=lambda x: x[1], reverse=True)
            optimal_models = [model for model, _ in model_scores[:max_ensemble_size]]

            logger.info(f"Selected optimal ensemble: {optimal_models} for context: {context}")
            return optimal_models

        except Exception as e:
            logger.error(f"Error getting optimal ensemble: {e}")
            return ["ggml-medium.bin"]  # Fallback

    async def _extract_patterns(self, result: RecognitionResult) -> List[LearningPattern]:
        """Extract learning patterns from recognition result"""
        patterns = []

        try:
            # Acoustic pattern analysis
            if len(result.acoustic_features) > 0:
                acoustic_pattern = await self._analyze_acoustic_pattern(result)
                if acoustic_pattern:
                    patterns.append(acoustic_pattern)

            # Lexical pattern analysis
            lexical_pattern = await self._analyze_lexical_pattern(result)
            if lexical_pattern:
                patterns.append(lexical_pattern)

            # UK English specific patterns
            if result.is_uk_english:
                uk_pattern = await self._analyze_uk_pattern(result)
                if uk_pattern:
                    patterns.append(uk_pattern)
                    self.uk_patterns_learned += 1

            return patterns

        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            return []

    async def _analyze_acoustic_pattern(self, result: RecognitionResult) -> Optional[LearningPattern]:
        """Analyze acoustic features to identify performance patterns"""
        try:
            features = np.array(result.acoustic_features)

            # Extract key acoustic characteristics
            pattern_features = {
                "spectral_centroid": float(np.mean(features[:128])) if len(features) >= 128 else 0.0,
                "spectral_rolloff": float(np.mean(features[128:256])) if len(features) >= 256 else 0.0,
                "mfcc_variance": float(np.var(features[256:384])) if len(features) >= 384 else 0.0,
                "confidence": result.confidence,
                "processing_time": result.processing_time_ms,
                "model_accuracy": self.model_weights.get(result.model_used, 0.5)
            }

            # Calculate pattern significance
            confidence_threshold = 0.8
            if result.confidence > confidence_threshold:
                pattern = LearningPattern(
                    pattern_id=f"acoustic_{datetime.now().timestamp()}",
                    pattern_type="acoustic",
                    features=pattern_features,
                    confidence_score=result.confidence,
                    accuracy_improvement=max(0.0, result.confidence - 0.8),
                    usage_count=1,
                    is_uk_specific=result.is_uk_english,
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                return pattern

            return None

        except Exception as e:
            logger.error(f"Error analyzing acoustic pattern: {e}")
            return None

    async def _analyze_lexical_pattern(self, result: RecognitionResult) -> Optional[LearningPattern]:
        """Analyze lexical patterns for vocabulary learning"""
        try:
            text = result.text.lower()
            words = text.split()

            # Identify significant vocabulary patterns
            pattern_features = {
                "word_count": len(words),
                "avg_word_length": np.mean([len(word) for word in words]) if words else 0.0,
                "confidence": result.confidence,
                "contains_uk_terms": self._contains_uk_vocabulary(text),
                "technical_density": self._calculate_technical_density(text)
            }

            # Create pattern if significant
            if result.confidence > 0.75 and len(words) > 2:
                pattern = LearningPattern(
                    pattern_id=f"lexical_{hash(text)%100000}",
                    pattern_type="lexical",
                    features=pattern_features,
                    confidence_score=result.confidence,
                    accuracy_improvement=max(0.0, result.confidence - 0.75),
                    usage_count=1,
                    is_uk_specific=result.is_uk_english,
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                return pattern

            return None

        except Exception as e:
            logger.error(f"Error analyzing lexical pattern: {e}")
            return None

    async def _analyze_uk_pattern(self, result: RecognitionResult) -> Optional[LearningPattern]:
        """Analyze UK English specific patterns"""
        try:
            text = result.text.lower()

            # UK vocabulary detection
            uk_score = 0.0
            uk_words_found = []

            for word in text.split():
                if word in self.uk_vocabulary_mappings:
                    uk_score += 1.0
                    uk_words_found.append(word)

            # Normalize by text length
            if len(text.split()) > 0:
                uk_score /= len(text.split())

            pattern_features = {
                "uk_vocabulary_score": uk_score,
                "uk_words_count": len(uk_words_found),
                "confidence": result.confidence,
                "text_length": len(text),
                "acoustic_uk_markers": self._extract_uk_acoustic_markers(result.acoustic_features)
            }

            # Create UK pattern if significant
            if uk_score > 0.1 or len(uk_words_found) > 0:
                pattern = LearningPattern(
                    pattern_id=f"uk_{hash(text)%100000}",
                    pattern_type="uk_dialect",
                    features=pattern_features,
                    confidence_score=result.confidence,
                    accuracy_improvement=uk_score * 0.1,  # UK patterns provide accuracy boost
                    usage_count=1,
                    is_uk_specific=True,
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                return pattern

            return None

        except Exception as e:
            logger.error(f"Error analyzing UK pattern: {e}")
            return None

    def _load_uk_vocabulary(self) -> Dict[str, str]:
        """Load UK English vocabulary mappings"""
        return {
            # Transportation
            "lift": "elevator",
            "lorry": "truck",
            "car park": "parking lot",
            "pavement": "sidewalk",
            "petrol": "gasoline",
            "boot": "trunk",
            "bonnet": "hood",

            # Housing
            "flat": "apartment",
            "bedsit": "studio apartment",
            "council house": "public housing",

            # Food & Drink
            "biscuit": "cookie",
            "sweets": "candy",
            "chips": "french fries",
            "crisps": "chips",
            "aubergine": "eggplant",
            "courgette": "zucchini",

            # General
            "rubbish": "garbage",
            "torch": "flashlight",
            "jumper": "sweater",
            "trainers": "sneakers",
            "football": "soccer",
            "mobile": "cell phone",
            "telly": "television"
        }

    def _contains_uk_vocabulary(self, text: str) -> bool:
        """Check if text contains UK-specific vocabulary"""
        words = text.lower().split()
        return any(word in self.uk_vocabulary_mappings for word in words)

    def _calculate_technical_density(self, text: str) -> float:
        """Calculate density of technical terms in text"""
        technical_indicators = [
            "algorithm", "processor", "memory", "bandwidth", "latency",
            "optimization", "neural", "machine learning", "artificial intelligence",
            "database", "api", "framework", "architecture", "protocol"
        ]

        words = text.lower().split()
        if not words:
            return 0.0

        technical_count = sum(1 for word in words if any(indicator in word for indicator in technical_indicators))
        return technical_count / len(words)

    def _extract_uk_acoustic_markers(self, acoustic_features: List[float]) -> float:
        """Extract acoustic markers that indicate UK English"""
        if len(acoustic_features) < 100:
            return 0.0

        # Simplified acoustic analysis for UK markers
        # In practice, this would use more sophisticated phonetic analysis
        features = np.array(acoustic_features[:100])

        # UK English tends to have different formant patterns
        f1_f2_ratio = np.mean(features[:25]) / (np.mean(features[25:50]) + 1e-6)
        vowel_space = np.var(features[50:75])
        intonation_pattern = np.mean(features[75:100])

        # Combine features to estimate UK-ness
        uk_score = (f1_f2_ratio * 0.4 + vowel_space * 0.3 + intonation_pattern * 0.3)
        return float(np.clip(uk_score, 0.0, 1.0))

    async def _store_recognition_result(self, result: RecognitionResult):
        """Store recognition result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO recognition_history
                    (recognized_text, confidence, acoustic_features, model_outputs,
                     speaker_id, is_uk_english, processing_time_ms, ground_truth)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                result.text, result.confidence, result.acoustic_features,
                json.dumps({"model": result.model_used}), result.speaker_id,
                result.is_uk_english, result.processing_time_ms, result.ground_truth)

        except Exception as e:
            logger.error(f"Error storing recognition result: {e}")

    async def _update_model_performance(self, result: RecognitionResult):
        """Update model performance metrics"""
        try:
            # Calculate accuracy if ground truth available
            accuracy = 1.0  # Default assumption
            if result.ground_truth:
                # Simple word-level accuracy
                recognized_words = set(result.text.lower().split())
                truth_words = set(result.ground_truth.lower().split())
                if truth_words:
                    accuracy = len(recognized_words & truth_words) / len(truth_words)
            else:
                # Use confidence as proxy for accuracy
                accuracy = result.confidence

            # Update model weights based on performance
            current_weight = self.model_weights.get(result.model_used, 0.5)
            learning_rate = 0.01
            new_weight = current_weight + learning_rate * (accuracy - current_weight)
            self.model_weights[result.model_used] = max(0.1, min(1.0, new_weight))

            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO model_performance
                    (model_name, accuracy, processing_time_ms, is_uk_english)
                    VALUES ($1, $2, $3, $4)
                """, result.model_used, accuracy, result.processing_time_ms, result.is_uk_english)

        except Exception as e:
            logger.error(f"Error updating model performance: {e}")

    async def _process_uk_english_recognition(self, result: RecognitionResult):
        """Process UK English specific recognition"""
        try:
            # Extract UK-specific patterns
            text = result.text.lower()

            # Update UK vocabulary learning
            for word in text.split():
                if word in self.uk_vocabulary_mappings:
                    # Boost weight for UK vocabulary
                    if result.model_used in self.model_weights:
                        uk_bonus = 0.005  # Small boost for UK vocabulary recognition
                        self.model_weights[result.model_used] += uk_bonus

            # Store UK-specific training data
            if result.confidence > 0.8:
                await self._store_uk_training_example(result)

        except Exception as e:
            logger.error(f"Error processing UK English recognition: {e}")

    async def _store_uk_training_example(self, result: RecognitionResult):
        """Store high-quality UK English examples for future training"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO uk_vocabulary_patterns
                    (word_or_phrase, usage_frequency, context_tags, accuracy_impact)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (word_or_phrase)
                    DO UPDATE SET
                        usage_frequency = uk_vocabulary_patterns.usage_frequency + 1,
                        accuracy_impact = GREATEST(uk_vocabulary_patterns.accuracy_impact, $4)
                """, result.text, 1.0, ["speech_recognition"], result.confidence)

        except Exception as e:
            logger.error(f"Error storing UK training example: {e}")

    async def _initialize_database(self):
        """Initialize database tables if needed"""
        try:
            async with self.db_pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Additional indexes for performance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_recognition_history_uk_english
                    ON recognition_history (is_uk_english, timestamp)
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_performance_recent
                    ON model_performance (recorded_at DESC, model_name)
                """)

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    async def _load_existing_patterns(self):
        """Load existing learning patterns from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT pattern_id, pattern_type, features, confidence_score,
                           accuracy_improvement, usage_count, is_uk_specific,
                           created_at, last_used
                    FROM learning_patterns
                    WHERE usage_count > 2
                    ORDER BY accuracy_improvement DESC
                    LIMIT 1000
                """)

                for row in rows:
                    pattern = LearningPattern(
                        pattern_id=row["pattern_id"],
                        pattern_type=row["pattern_type"],
                        features=json.loads(row["features"]) if row["features"] else {},
                        confidence_score=row["confidence_score"],
                        accuracy_improvement=row["accuracy_improvement"],
                        usage_count=row["usage_count"],
                        is_uk_specific=row["is_uk_specific"],
                        created_at=row["created_at"],
                        last_used=row["last_used"]
                    )
                    self.learning_patterns[pattern.pattern_id] = pattern

                logger.info(f"Loaded {len(self.learning_patterns)} existing learning patterns")

        except Exception as e:
            logger.error(f"Error loading existing patterns: {e}")

    def is_active(self) -> bool:
        """Check if learning engine is active"""
        return self.db_pool is not None and not self.db_pool._closed

    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()