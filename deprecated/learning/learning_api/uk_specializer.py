"""
UK English Specializer for VoiceStand Learning System
Advanced UK dialect recognition and optimization
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
import asyncpg
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class UKDialectFeatures:
    rhoticity_score: float = 0.0  # R-dropping characteristics
    vowel_system_score: float = 0.0  # UK vowel system detection
    lexical_choice_score: float = 0.0  # British vs American vocabulary
    intonation_score: float = 0.0  # British intonation patterns
    overall_uk_probability: float = 0.0

@dataclass
class UKVocabularyMapping:
    american_term: str
    british_term: str
    confidence: float
    usage_frequency: int
    context_tags: List[str]
    accuracy_impact: float

class UKEnglishSpecializer:
    """Advanced UK English dialect specialization and optimization"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_pool = None

        # UK vocabulary mappings (comprehensive)
        self.uk_vocabulary = self._initialize_uk_vocabulary()

        # UK pronunciation patterns
        self.uk_pronunciation_patterns = self._initialize_pronunciation_patterns()

        # UK-specific linguistic markers
        self.uk_linguistic_markers = self._initialize_linguistic_markers()

        # Accent detection patterns
        self.accent_detection_features = self._initialize_accent_features()

        # Performance tracking
        self.uk_recognition_stats = {
            "total_processed": 0,
            "high_confidence_count": 0,
            "vocabulary_corrections": 0,
            "accent_detections": 0
        }

    async def initialize(self):
        """Initialize UK specializer with database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=8,
                command_timeout=60
            )
            await self._load_existing_uk_patterns()
            logger.info("UK English specializer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize UK specializer: {e}")
            return False

    async def analyze_uk_pattern(self, text: str, audio_features: List[float], confidence: float) -> Dict[str, Any]:
        """Analyze text and audio for UK English patterns"""
        try:
            # Extract UK dialect features
            dialect_features = self._extract_dialect_features(text, audio_features)

            # Analyze vocabulary usage
            vocabulary_analysis = self._analyze_uk_vocabulary(text)

            # Calculate overall UK probability
            uk_probability = self._calculate_uk_probability(dialect_features, vocabulary_analysis)

            # Generate recommendations
            recommendations = await self._generate_uk_recommendations(
                text, dialect_features, vocabulary_analysis, confidence
            )

            # Update statistics
            self.uk_recognition_stats["total_processed"] += 1
            if confidence > 0.85:
                self.uk_recognition_stats["high_confidence_count"] += 1

            # Store pattern if significant
            if uk_probability > 0.3 or vocabulary_analysis["uk_words_found"]:
                await self._store_uk_pattern(text, audio_features, dialect_features, vocabulary_analysis)

            return {
                "uk_probability": uk_probability,
                "dialect_features": dialect_features.__dict__,
                "vocabulary_analysis": vocabulary_analysis,
                "recommendations": recommendations,
                "confidence_boost": max(0.0, uk_probability * 0.1)  # Boost for UK recognition
            }

        except Exception as e:
            logger.error(f"Error analyzing UK pattern: {e}")
            return {"uk_probability": 0.0, "recommendations": []}

    async def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Get UK English specific improvement suggestions"""
        suggestions = []

        try:
            async with self.db_pool.acquire() as conn:
                # Analyze UK vocabulary performance
                vocab_performance = await conn.fetch("""
                    SELECT word_or_phrase, usage_frequency, accuracy_impact,
                           american_variant, british_variant
                    FROM uk_vocabulary_patterns
                    WHERE usage_frequency > 3
                    ORDER BY accuracy_impact ASC
                    LIMIT 20
                """)

                if vocab_performance:
                    low_performing_words = [
                        row for row in vocab_performance
                        if row['accuracy_impact'] < 0.7
                    ]

                    if low_performing_words:
                        suggestions.append({
                            "insight_type": "uk_vocabulary_improvement",
                            "description": f"Identified {len(low_performing_words)} UK vocabulary items with low accuracy",
                            "confidence": 0.85,
                            "recommendations": {
                                "action": "targeted_uk_vocabulary_training",
                                "low_performing_words": [
                                    {
                                        "word": row['word_or_phrase'],
                                        "american": row['american_variant'],
                                        "british": row['british_variant'],
                                        "accuracy": row['accuracy_impact']
                                    }
                                    for row in low_performing_words[:10]
                                ],
                                "training_method": "contextual_fine_tuning"
                            },
                            "requires_retraining": True
                        })

                # Analyze UK accent detection performance
                accent_stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_uk_samples,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN confidence > 0.9 THEN 1 END) as high_conf_count
                    FROM recognition_history
                    WHERE is_uk_english = true
                    AND timestamp > NOW() - INTERVAL '7 days'
                """)

                if accent_stats and accent_stats['total_uk_samples'] > 20:
                    uk_high_conf_ratio = (accent_stats['high_conf_count'] / accent_stats['total_uk_samples'])

                    if uk_high_conf_ratio < 0.6:  # Less than 60% high confidence
                        suggestions.append({
                            "insight_type": "uk_accent_recognition_improvement",
                            "description": f"UK accent recognition needs improvement "
                                         f"(only {uk_high_conf_ratio:.1%} high confidence)",
                            "confidence": 0.9,
                            "recommendations": {
                                "action": "uk_accent_training",
                                "current_performance": float(uk_high_conf_ratio),
                                "target_performance": 0.8,
                                "training_focus": "acoustic_modeling"
                            },
                            "requires_retraining": True
                        })

                # Check for underrepresented UK regions/accents
                suggestions.extend(await self._analyze_regional_coverage())

        except Exception as e:
            logger.error(f"Error getting UK improvement suggestions: {e}")

        return suggestions

    async def get_vocabulary_suggestions(self, text: str) -> List[Dict[str, str]]:
        """Get UK vocabulary suggestions for given text"""
        suggestions = []

        try:
            words = text.lower().split()
            text_lower = text.lower()

            # Check for American terms that have UK equivalents
            for american_term, uk_mapping in self.uk_vocabulary.items():
                if american_term in text_lower:
                    suggestions.append({
                        "original": american_term,
                        "uk_alternative": uk_mapping["uk_term"],
                        "confidence": uk_mapping["confidence"],
                        "context": uk_mapping["context"]
                    })

            # Check for partial matches (phrases)
            for american_phrase in self.uk_vocabulary:
                if len(american_phrase.split()) > 1:  # Multi-word phrases
                    if american_phrase in text_lower:
                        uk_mapping = self.uk_vocabulary[american_phrase]
                        suggestions.append({
                            "original": american_phrase,
                            "uk_alternative": uk_mapping["uk_term"],
                            "confidence": uk_mapping["confidence"],
                            "context": uk_mapping["context"]
                        })

            # Sort by confidence
            suggestions.sort(key=lambda x: x["confidence"], reverse=True)

            return suggestions[:10]  # Return top 10 suggestions

        except Exception as e:
            logger.error(f"Error getting vocabulary suggestions: {e}")
            return []

    def _extract_dialect_features(self, text: str, audio_features: List[float]) -> UKDialectFeatures:
        """Extract UK dialect features from text and audio"""
        features = UKDialectFeatures()

        try:
            # Lexical analysis
            features.lexical_choice_score = self._calculate_lexical_uk_score(text)

            # Audio feature analysis (simplified acoustic modeling)
            if len(audio_features) >= 128:
                features.rhoticity_score = self._analyze_rhoticity(audio_features)
                features.vowel_system_score = self._analyze_vowel_system(audio_features)
                features.intonation_score = self._analyze_intonation(audio_features)

            # Calculate overall probability
            features.overall_uk_probability = (
                features.lexical_choice_score * 0.4 +
                features.rhoticity_score * 0.25 +
                features.vowel_system_score * 0.25 +
                features.intonation_score * 0.1
            )

        except Exception as e:
            logger.error(f"Error extracting dialect features: {e}")

        return features

    def _analyze_uk_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze UK vocabulary usage in text"""
        text_lower = text.lower()
        words = text_lower.split()

        uk_words_found = []
        american_words_found = []
        total_vocabulary_score = 0.0

        # Check for UK-specific terms
        for word in words:
            if word in self.uk_vocabulary:
                uk_mapping = self.uk_vocabulary[word]
                uk_words_found.append({
                    "word": word,
                    "uk_term": uk_mapping["uk_term"],
                    "confidence": uk_mapping["confidence"]
                })
                total_vocabulary_score += uk_mapping["confidence"]

        # Check for American terms that should be UK
        for american_term, uk_mapping in self.uk_vocabulary.items():
            if american_term in text_lower and american_term not in [w["word"] for w in uk_words_found]:
                american_words_found.append({
                    "american_term": american_term,
                    "suggested_uk_term": uk_mapping["uk_term"],
                    "confidence": uk_mapping["confidence"]
                })

        # Normalize score
        if words:
            total_vocabulary_score /= len(words)

        return {
            "uk_words_found": uk_words_found,
            "american_words_found": american_words_found,
            "vocabulary_score": total_vocabulary_score,
            "uk_word_count": len(uk_words_found),
            "total_words": len(words)
        }

    def _calculate_lexical_uk_score(self, text: str) -> float:
        """Calculate lexical UK score based on vocabulary usage"""
        text_lower = text.lower()
        words = text_lower.split()

        if not words:
            return 0.0

        uk_indicators = 0
        total_indicators = 0

        # Check for UK-specific vocabulary
        for word in words:
            if word in self.uk_vocabulary:
                uk_indicators += self.uk_vocabulary[word]["confidence"]
                total_indicators += 1

        # Check for UK-specific phrases and expressions
        for phrase, score in self.uk_linguistic_markers.items():
            if phrase in text_lower:
                uk_indicators += score
                total_indicators += 1

        return min(1.0, uk_indicators / max(1, total_indicators))

    def _analyze_rhoticity(self, audio_features: List[float]) -> float:
        """Analyze rhoticity (R-dropping) characteristics"""
        try:
            # Simplified rhoticity analysis based on spectral features
            # In practice, this would use more sophisticated phonetic analysis
            if len(audio_features) < 64:
                return 0.0

            # R-sounds typically appear in specific frequency ranges
            r_sound_features = audio_features[32:48]  # Simplified representation
            r_intensity = np.mean(r_sound_features)
            r_variance = np.var(r_sound_features)

            # UK English typically has less rhotic pronunciation
            # Lower intensity and variance suggest non-rhotic accent (UK characteristic)
            rhoticity_score = 1.0 - min(1.0, (r_intensity * 2 + r_variance))

            return max(0.0, rhoticity_score)

        except Exception as e:
            logger.error(f"Error analyzing rhoticity: {e}")
            return 0.0

    def _analyze_vowel_system(self, audio_features: List[float]) -> float:
        """Analyze vowel system characteristics"""
        try:
            if len(audio_features) < 96:
                return 0.0

            # UK English has distinctive vowel patterns
            vowel_features = audio_features[48:80]

            # Analyze formant patterns (simplified)
            f1_pattern = np.mean(vowel_features[:16])
            f2_pattern = np.mean(vowel_features[16:32])

            # UK English typically has different vowel space
            uk_vowel_score = abs(f1_pattern - 0.5) + abs(f2_pattern - 0.6)

            return min(1.0, uk_vowel_score)

        except Exception as e:
            logger.error(f"Error analyzing vowel system: {e}")
            return 0.0

    def _analyze_intonation(self, audio_features: List[float]) -> float:
        """Analyze intonation patterns"""
        try:
            if len(audio_features) < 128:
                return 0.0

            # UK English has distinctive intonation patterns
            pitch_features = audio_features[80:112]

            # Analyze pitch contour characteristics
            pitch_variance = np.var(pitch_features)
            pitch_trend = np.polyfit(range(len(pitch_features)), pitch_features, 1)[0]

            # UK English often has rising intonation in statements
            uk_intonation_score = min(1.0, abs(pitch_trend) + pitch_variance * 0.5)

            return uk_intonation_score

        except Exception as e:
            logger.error(f"Error analyzing intonation: {e}")
            return 0.0

    def _calculate_uk_probability(self, dialect_features: UKDialectFeatures, vocabulary_analysis: Dict[str, Any]) -> float:
        """Calculate overall UK English probability"""
        try:
            # Combine multiple evidence sources
            acoustic_score = (
                dialect_features.rhoticity_score * 0.3 +
                dialect_features.vowel_system_score * 0.3 +
                dialect_features.intonation_score * 0.2
            )

            lexical_score = vocabulary_analysis["vocabulary_score"]

            # Additional boost for explicit UK vocabulary
            vocabulary_boost = min(0.3, vocabulary_analysis["uk_word_count"] * 0.1)

            # Combine scores with weights
            total_score = acoustic_score * 0.6 + lexical_score * 0.3 + vocabulary_boost

            return min(1.0, total_score)

        except Exception as e:
            logger.error(f"Error calculating UK probability: {e}")
            return 0.0

    async def _generate_uk_recommendations(self, text: str, dialect_features: UKDialectFeatures,
                                         vocabulary_analysis: Dict[str, Any], confidence: float) -> List[Dict[str, Any]]:
        """Generate UK English specific recommendations"""
        recommendations = []

        try:
            # Vocabulary recommendations
            if vocabulary_analysis["american_words_found"]:
                recommendations.append({
                    "type": "vocabulary_correction",
                    "description": f"Found {len(vocabulary_analysis['american_words_found'])} American terms",
                    "suggestions": vocabulary_analysis["american_words_found"][:3],
                    "priority": "medium"
                })

            # Acoustic recommendations
            if dialect_features.overall_uk_probability > 0.5 and confidence < 0.8:
                recommendations.append({
                    "type": "acoustic_optimization",
                    "description": "UK accent detected but low confidence - consider acoustic model tuning",
                    "uk_probability": dialect_features.overall_uk_probability,
                    "priority": "high"
                })

            # Training data recommendations
            if dialect_features.overall_uk_probability > 0.7:
                recommendations.append({
                    "type": "training_data_collection",
                    "description": "High-quality UK English sample - suitable for training data",
                    "quality_score": dialect_features.overall_uk_probability,
                    "priority": "low"
                })

        except Exception as e:
            logger.error(f"Error generating UK recommendations: {e}")

        return recommendations

    async def _store_uk_pattern(self, text: str, audio_features: List[float],
                              dialect_features: UKDialectFeatures, vocabulary_analysis: Dict[str, Any]):
        """Store UK pattern in database for learning"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store in learning patterns table
                pattern_features = {
                    "rhoticity_score": dialect_features.rhoticity_score,
                    "vowel_system_score": dialect_features.vowel_system_score,
                    "lexical_choice_score": dialect_features.lexical_choice_score,
                    "intonation_score": dialect_features.intonation_score,
                    "uk_probability": dialect_features.overall_uk_probability,
                    "uk_word_count": vocabulary_analysis["uk_word_count"],
                    "vocabulary_score": vocabulary_analysis["vocabulary_score"]
                }

                await conn.execute("""
                    INSERT INTO learning_patterns
                    (pattern_id, pattern_type, features, confidence_score,
                     accuracy_improvement, usage_count, is_uk_specific)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (pattern_id) DO UPDATE SET
                        usage_count = learning_patterns.usage_count + 1,
                        last_used = NOW()
                """,
                f"uk_pattern_{hash(text) % 100000}",
                "uk_dialect",
                json.dumps(pattern_features),
                dialect_features.overall_uk_probability,
                dialect_features.overall_uk_probability * 0.1,  # Potential accuracy improvement
                1,
                True)

                # Update vocabulary patterns
                for uk_word in vocabulary_analysis["uk_words_found"]:
                    await conn.execute("""
                        INSERT INTO uk_vocabulary_patterns
                        (word_or_phrase, usage_frequency, accuracy_impact)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (word_or_phrase) DO UPDATE SET
                            usage_frequency = uk_vocabulary_patterns.usage_frequency + 1,
                            accuracy_impact = GREATEST(uk_vocabulary_patterns.accuracy_impact, $3)
                    """,
                    uk_word["word"],
                    1,
                    uk_word["confidence"])

        except Exception as e:
            logger.error(f"Error storing UK pattern: {e}")

    async def _analyze_regional_coverage(self) -> List[Dict[str, Any]]:
        """Analyze coverage of different UK regional accents"""
        suggestions = []

        try:
            # This would typically analyze regional accent patterns
            # For now, provide general recommendations

            suggestions.append({
                "insight_type": "regional_accent_coverage",
                "description": "Consider expanding training data to include more UK regional accents",
                "confidence": 0.7,
                "recommendations": {
                    "action": "regional_data_collection",
                    "target_regions": ["Scottish", "Welsh", "Northern England", "West Country"],
                    "current_coverage": "limited",
                    "priority": "medium"
                },
                "requires_retraining": False
            })

        except Exception as e:
            logger.error(f"Error analyzing regional coverage: {e}")

        return suggestions

    async def _load_existing_uk_patterns(self):
        """Load existing UK patterns from database"""
        try:
            async with self.db_pool.acquire() as conn:
                patterns = await conn.fetch("""
                    SELECT pattern_id, features, confidence_score, usage_count
                    FROM learning_patterns
                    WHERE is_uk_specific = true
                    AND usage_count > 2
                    ORDER BY confidence_score DESC
                    LIMIT 500
                """)

                logger.info(f"Loaded {len(patterns)} existing UK patterns")

        except Exception as e:
            logger.error(f"Error loading existing UK patterns: {e}")

    def _initialize_uk_vocabulary(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive UK vocabulary mappings"""
        return {
            # Transportation
            "elevator": {"uk_term": "lift", "confidence": 0.95, "context": "building"},
            "truck": {"uk_term": "lorry", "confidence": 0.9, "context": "transport"},
            "parking lot": {"uk_term": "car park", "confidence": 0.95, "context": "transport"},
            "sidewalk": {"uk_term": "pavement", "confidence": 0.9, "context": "urban"},
            "gasoline": {"uk_term": "petrol", "confidence": 0.95, "context": "automotive"},
            "hood": {"uk_term": "bonnet", "confidence": 0.85, "context": "automotive"},
            "trunk": {"uk_term": "boot", "confidence": 0.85, "context": "automotive"},
            "fender": {"uk_term": "wing", "confidence": 0.8, "context": "automotive"},
            "turn signal": {"uk_term": "indicator", "confidence": 0.85, "context": "automotive"},

            # Housing
            "apartment": {"uk_term": "flat", "confidence": 0.9, "context": "housing"},
            "studio apartment": {"uk_term": "bedsit", "confidence": 0.8, "context": "housing"},
            "duplex": {"uk_term": "semi-detached", "confidence": 0.8, "context": "housing"},
            "yard": {"uk_term": "garden", "confidence": 0.75, "context": "housing"},
            "mailbox": {"uk_term": "postbox", "confidence": 0.85, "context": "housing"},

            # Food & Drink
            "cookie": {"uk_term": "biscuit", "confidence": 0.9, "context": "food"},
            "candy": {"uk_term": "sweets", "confidence": 0.9, "context": "food"},
            "french fries": {"uk_term": "chips", "confidence": 0.95, "context": "food"},
            "chips": {"uk_term": "crisps", "confidence": 0.9, "context": "food"},
            "eggplant": {"uk_term": "aubergine", "confidence": 0.85, "context": "food"},
            "zucchini": {"uk_term": "courgette", "confidence": 0.85, "context": "food"},
            "cilantro": {"uk_term": "coriander", "confidence": 0.8, "context": "food"},
            "arugula": {"uk_term": "rocket", "confidence": 0.8, "context": "food"},
            "aluminum foil": {"uk_term": "tin foil", "confidence": 0.8, "context": "kitchen"},

            # Clothing
            "sweater": {"uk_term": "jumper", "confidence": 0.85, "context": "clothing"},
            "sneakers": {"uk_term": "trainers", "confidence": 0.9, "context": "clothing"},
            "pants": {"uk_term": "trousers", "confidence": 0.85, "context": "clothing"},
            "undershirt": {"uk_term": "vest", "confidence": 0.8, "context": "clothing"},
            "suspenders": {"uk_term": "braces", "confidence": 0.8, "context": "clothing"},

            # General items
            "garbage": {"uk_term": "rubbish", "confidence": 0.9, "context": "waste"},
            "trash can": {"uk_term": "bin", "confidence": 0.85, "context": "waste"},
            "flashlight": {"uk_term": "torch", "confidence": 0.9, "context": "tools"},
            "eraser": {"uk_term": "rubber", "confidence": 0.85, "context": "stationery"},
            "band-aid": {"uk_term": "plaster", "confidence": 0.8, "context": "medical"},

            # Technology
            "cell phone": {"uk_term": "mobile", "confidence": 0.9, "context": "technology"},
            "tv": {"uk_term": "telly", "confidence": 0.7, "context": "technology"},
            "vacation": {"uk_term": "holiday", "confidence": 0.85, "context": "leisure"},

            # Sports
            "soccer": {"uk_term": "football", "confidence": 0.95, "context": "sports"},

            # Education
            "elementary school": {"uk_term": "primary school", "confidence": 0.9, "context": "education"},
            "high school": {"uk_term": "secondary school", "confidence": 0.85, "context": "education"},
            "college": {"uk_term": "university", "confidence": 0.7, "context": "education"},
            "math": {"uk_term": "maths", "confidence": 0.8, "context": "education"}
        }

    def _initialize_pronunciation_patterns(self) -> Dict[str, List[str]]:
        """Initialize UK pronunciation pattern variants"""
        return {
            "schedule": ["shed-yool", "sked-yool"],
            "privacy": ["priv-a-see", "pry-va-see"],
            "lever": ["lee-ver"],
            "advertisement": ["ad-vert-is-ment"],
            "laboratory": ["la-bor-a-tree"],
            "aluminium": ["al-yoo-min-ee-um"]
        }

    def _initialize_linguistic_markers(self) -> Dict[str, float]:
        """Initialize UK linguistic markers and their weights"""
        return {
            "brilliant": 0.7,
            "lovely": 0.6,
            "quite good": 0.8,
            "rather": 0.7,
            "bloody": 0.9,
            "bloke": 0.9,
            "mate": 0.8,
            "cheers": 0.8,
            "right then": 0.8,
            "proper": 0.6,
            "spot on": 0.9,
            "chap": 0.8,
            "queue": 0.9,
            "whilst": 0.8,
            "amongst": 0.7,
            "colour": 0.9,
            "favour": 0.9,
            "honour": 0.9,
            "centre": 0.9,
            "theatre": 0.9,
            "organise": 0.8,
            "realise": 0.8,
            "analyse": 0.8
        }

    def _initialize_accent_features(self) -> Dict[str, Any]:
        """Initialize accent detection feature patterns"""
        return {
            "rp_features": {
                "non_rhotic": 0.9,
                "long_vowels": 0.8,
                "clear_consonants": 0.85
            },
            "northern_features": {
                "flat_vowels": 0.8,
                "short_a": 0.9
            },
            "scottish_features": {
                "rhotic": 0.9,
                "rolled_r": 0.85
            }
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()