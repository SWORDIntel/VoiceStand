"""
Pattern Analyzer for VoiceStand Learning System
Advanced pattern recognition for continuous improvement
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import asyncpg

logger = logging.getLogger(__name__)

@dataclass
class PatternInsight:
    insight_type: str
    description: str
    confidence: float
    recommendations: Dict[str, Any]
    requires_retraining: bool = False
    affected_models: List[str] = None
    potential_accuracy_gain: float = 0.0

class PatternAnalyzer:
    """Advanced pattern analysis for identifying improvement opportunities"""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_pool = None

        # Pattern analysis configuration
        self.min_pattern_frequency = 5
        self.confidence_threshold = 0.8
        self.similarity_threshold = 0.85

        # UK English specialization
        self.uk_pattern_weight = 1.3  # Higher weight for UK patterns

        # Clustering parameters
        self.clustering_eps = 0.3
        self.min_cluster_size = 3

    async def initialize(self):
        """Initialize pattern analyzer with database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Pattern analyzer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pattern analyzer: {e}")
            return False

    async def analyze_patterns(self) -> List[PatternInsight]:
        """Analyze all patterns and generate insights"""
        insights = []

        try:
            # Analyze acoustic patterns
            acoustic_insights = await self._analyze_acoustic_patterns()
            insights.extend(acoustic_insights)

            # Analyze lexical patterns
            lexical_insights = await self._analyze_lexical_patterns()
            insights.extend(lexical_insights)

            # Analyze UK English patterns
            uk_insights = await self._analyze_uk_patterns()
            insights.extend(uk_insights)

            # Analyze model performance patterns
            performance_insights = await self._analyze_performance_patterns()
            insights.extend(performance_insights)

            # Analyze ensemble effectiveness
            ensemble_insights = await self._analyze_ensemble_patterns()
            insights.extend(ensemble_insights)

            # Sort by potential accuracy gain
            insights.sort(key=lambda x: x.potential_accuracy_gain, reverse=True)

            logger.info(f"Generated {len(insights)} pattern insights")
            return insights

        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return []

    async def _analyze_acoustic_patterns(self) -> List[PatternInsight]:
        """Analyze acoustic feature patterns for optimization opportunities"""
        insights = []

        try:
            async with self.db_pool.acquire() as conn:
                # Get recent acoustic data
                rows = await conn.fetch("""
                    SELECT acoustic_features, confidence, is_uk_english, model_outputs
                    FROM recognition_history
                    WHERE timestamp > NOW() - INTERVAL '7 days'
                    AND acoustic_features IS NOT NULL
                    AND array_length(acoustic_features, 1) > 100
                    ORDER BY timestamp DESC
                    LIMIT 5000
                """)

                if len(rows) < 50:
                    return insights

                # Extract feature vectors
                features = []
                confidences = []
                uk_flags = []

                for row in rows:
                    if row['acoustic_features'] and len(row['acoustic_features']) >= 128:
                        features.append(row['acoustic_features'][:128])  # Standardize feature size
                        confidences.append(row['confidence'])
                        uk_flags.append(row['is_uk_english'])

                if len(features) < 20:
                    return insights

                features_array = np.array(features)
                confidences_array = np.array(confidences)

                # Normalize features
                scaler = StandardScaler()
                features_normalized = scaler.fit_transform(features_array)

                # Cluster acoustic patterns
                clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_cluster_size)
                cluster_labels = clustering.fit_predict(features_normalized)

                # Analyze clusters
                unique_clusters = set(cluster_labels)
                unique_clusters.discard(-1)  # Remove noise cluster

                for cluster_id in unique_clusters:
                    cluster_mask = cluster_labels == cluster_id
                    cluster_confidences = confidences_array[cluster_mask]
                    cluster_uk_flags = [uk_flags[i] for i in range(len(uk_flags)) if cluster_mask[i]]

                    cluster_size = np.sum(cluster_mask)
                    avg_confidence = np.mean(cluster_confidences)
                    uk_ratio = np.mean(cluster_uk_flags) if cluster_uk_flags else 0.0

                    # Identify significant patterns
                    if cluster_size >= self.min_cluster_size:
                        if avg_confidence > 0.9:
                            # High-performance acoustic pattern
                            insights.append(PatternInsight(
                                insight_type="high_performance_acoustic",
                                description=f"Acoustic cluster {cluster_id} shows excellent performance "
                                          f"(avg confidence: {avg_confidence:.3f}, {cluster_size} samples)",
                                confidence=avg_confidence,
                                recommendations={
                                    "action": "optimize_for_pattern",
                                    "cluster_id": int(cluster_id),
                                    "feature_weights": features_normalized[cluster_mask].mean(axis=0).tolist()[:20],
                                    "uk_optimized": uk_ratio > 0.5
                                },
                                potential_accuracy_gain=min(0.05, (avg_confidence - 0.85) * 0.1),
                                affected_models=["all"]
                            ))

                        elif avg_confidence < 0.7:
                            # Low-performance acoustic pattern
                            insights.append(PatternInsight(
                                insight_type="low_performance_acoustic",
                                description=f"Acoustic cluster {cluster_id} shows poor performance "
                                          f"(avg confidence: {avg_confidence:.3f}, {cluster_size} samples)",
                                confidence=1.0 - avg_confidence,
                                recommendations={
                                    "action": "retrain_for_pattern",
                                    "cluster_id": int(cluster_id),
                                    "feature_characteristics": features_normalized[cluster_mask].mean(axis=0).tolist()[:20],
                                    "uk_specific": uk_ratio > 0.7
                                },
                                requires_retraining=True,
                                potential_accuracy_gain=min(0.15, (0.85 - avg_confidence) * 0.2),
                                affected_models=["ggml-small.bin", "ggml-medium.bin"]
                            ))

                # UK English specific acoustic analysis
                if any(uk_flags):
                    uk_features = features_array[np.array(uk_flags)]
                    uk_confidences = confidences_array[np.array(uk_flags)]

                    if len(uk_features) >= 10:
                        uk_avg_confidence = np.mean(uk_confidences)
                        non_uk_avg_confidence = np.mean(confidences_array[~np.array(uk_flags)])

                        if uk_avg_confidence < non_uk_avg_confidence - 0.05:
                            insights.append(PatternInsight(
                                insight_type="uk_acoustic_underperformance",
                                description=f"UK English acoustic patterns underperforming "
                                          f"(UK: {uk_avg_confidence:.3f} vs Non-UK: {non_uk_avg_confidence:.3f})",
                                confidence=0.9,
                                recommendations={
                                    "action": "uk_acoustic_optimization",
                                    "uk_feature_profile": uk_features.mean(axis=0).tolist()[:20],
                                    "training_priority": "high"
                                },
                                requires_retraining=True,
                                potential_accuracy_gain=min(0.08, (non_uk_avg_confidence - uk_avg_confidence) * 0.3),
                                affected_models=["uk-english-fine-tuned-small.bin", "uk-english-fine-tuned-medium.bin"]
                            ))

        except Exception as e:
            logger.error(f"Error analyzing acoustic patterns: {e}")

        return insights

    async def _analyze_lexical_patterns(self) -> List[PatternInsight]:
        """Analyze lexical patterns for vocabulary optimization"""
        insights = []

        try:
            async with self.db_pool.acquire() as conn:
                # Get recent recognition data with vocabulary analysis
                rows = await conn.fetch("""
                    SELECT recognized_text, confidence, is_uk_english, ground_truth
                    FROM recognition_history
                    WHERE timestamp > NOW() - INTERVAL '7 days'
                    AND recognized_text IS NOT NULL
                    AND LENGTH(recognized_text) > 5
                    ORDER BY timestamp DESC
                    LIMIT 3000
                """)

                if len(rows) < 50:
                    return insights

                # Analyze word-level performance
                word_performance = {}
                uk_word_performance = {}

                for row in rows:
                    text = row['recognized_text'].lower()
                    confidence = row['confidence']
                    is_uk = row['is_uk_english']

                    words = text.split()
                    for word in words:
                        if len(word) > 2:  # Skip very short words
                            if word not in word_performance:
                                word_performance[word] = []
                            word_performance[word].append(confidence)

                            if is_uk:
                                if word not in uk_word_performance:
                                    uk_word_performance[word] = []
                                uk_word_performance[word].append(confidence)

                # Identify problematic words
                problematic_words = []
                excellent_words = []

                for word, confidences in word_performance.items():
                    if len(confidences) >= 5:  # Sufficient data
                        avg_confidence = np.mean(confidences)

                        if avg_confidence < 0.7:
                            problematic_words.append((word, avg_confidence, len(confidences)))
                        elif avg_confidence > 0.95:
                            excellent_words.append((word, avg_confidence, len(confidences)))

                # Generate insights for problematic words
                if problematic_words:
                    problematic_words.sort(key=lambda x: x[1])  # Sort by confidence
                    top_problematic = problematic_words[:10]

                    insights.append(PatternInsight(
                        insight_type="low_performance_vocabulary",
                        description=f"Identified {len(problematic_words)} words with poor recognition performance",
                        confidence=0.85,
                        recommendations={
                            "action": "vocabulary_focused_training",
                            "problematic_words": [{"word": w, "confidence": c, "frequency": f}
                                                for w, c, f in top_problematic],
                            "training_type": "word_level_fine_tuning"
                        },
                        requires_retraining=True,
                        potential_accuracy_gain=min(0.12, len(problematic_words) * 0.001),
                        affected_models=["all"]
                    ))

                # UK-specific vocabulary analysis
                uk_specific_issues = []
                for word, confidences in uk_word_performance.items():
                    if len(confidences) >= 3:
                        uk_avg = np.mean(confidences)
                        general_avg = np.mean(word_performance.get(word, []))

                        if uk_avg < general_avg - 0.1:  # UK performance significantly worse
                            uk_specific_issues.append((word, uk_avg, general_avg))

                if uk_specific_issues:
                    insights.append(PatternInsight(
                        insight_type="uk_vocabulary_issues",
                        description=f"Identified {len(uk_specific_issues)} words with UK English recognition issues",
                        confidence=0.9,
                        recommendations={
                            "action": "uk_vocabulary_training",
                            "uk_problem_words": [{"word": w, "uk_conf": uk_c, "general_conf": g_c}
                                               for w, uk_c, g_c in uk_specific_issues[:10]],
                            "priority": "high"
                        },
                        requires_retraining=True,
                        potential_accuracy_gain=min(0.1, len(uk_specific_issues) * 0.002),
                        affected_models=["uk-english-fine-tuned-small.bin", "uk-english-fine-tuned-medium.bin"]
                    ))

        except Exception as e:
            logger.error(f"Error analyzing lexical patterns: {e}")

        return insights

    async def _analyze_uk_patterns(self) -> List[PatternInsight]:
        """Analyze UK English specific patterns"""
        insights = []

        try:
            async with self.db_pool.acquire() as conn:
                # Get UK English performance data
                uk_stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_uk_samples,
                        AVG(confidence) as avg_uk_confidence,
                        COUNT(CASE WHEN confidence > 0.9 THEN 1 END) as high_conf_count,
                        COUNT(CASE WHEN confidence < 0.7 THEN 1 END) as low_conf_count
                    FROM recognition_history
                    WHERE is_uk_english = true
                    AND timestamp > NOW() - INTERVAL '7 days'
                """)

                general_stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_samples,
                        AVG(confidence) as avg_confidence
                    FROM recognition_history
                    WHERE is_uk_english = false
                    AND timestamp > NOW() - INTERVAL '7 days'
                """)

                if uk_stats and general_stats and uk_stats['total_uk_samples'] > 10:
                    uk_confidence = uk_stats['avg_uk_confidence']
                    general_confidence = general_stats['avg_confidence']
                    confidence_gap = general_confidence - uk_confidence

                    if confidence_gap > 0.05:  # Significant performance gap
                        insights.append(PatternInsight(
                            insight_type="uk_performance_gap",
                            description=f"UK English recognition underperforming by {confidence_gap:.3f} "
                                      f"({uk_confidence:.3f} vs {general_confidence:.3f})",
                            confidence=0.95,
                            recommendations={
                                "action": "uk_model_enhancement",
                                "performance_gap": float(confidence_gap),
                                "uk_sample_count": uk_stats['total_uk_samples'],
                                "suggested_training_hours": max(10, confidence_gap * 100)
                            },
                            requires_retraining=True,
                            potential_accuracy_gain=min(0.15, confidence_gap * 0.5),
                            affected_models=["uk-english-fine-tuned-small.bin", "uk-english-fine-tuned-medium.bin"]
                        ))

                    # Analyze UK vocabulary usage patterns
                    uk_vocab_data = await conn.fetch("""
                        SELECT word_or_phrase, usage_frequency, accuracy_impact
                        FROM uk_vocabulary_patterns
                        WHERE usage_frequency > 2
                        ORDER BY accuracy_impact DESC
                        LIMIT 50
                    """)

                    if uk_vocab_data:
                        high_impact_words = [row for row in uk_vocab_data if row['accuracy_impact'] > 0.8]
                        low_impact_words = [row for row in uk_vocab_data if row['accuracy_impact'] < 0.6]

                        if high_impact_words:
                            insights.append(PatternInsight(
                                insight_type="uk_vocabulary_strengths",
                                description=f"Identified {len(high_impact_words)} high-performing UK vocabulary items",
                                confidence=0.9,
                                recommendations={
                                    "action": "expand_uk_vocabulary_training",
                                    "high_impact_words": [{"word": row['word_or_phrase'],
                                                         "impact": row['accuracy_impact']}
                                                        for row in high_impact_words[:10]],
                                    "training_strategy": "similar_word_expansion"
                                },
                                potential_accuracy_gain=0.03,
                                affected_models=["uk-english-fine-tuned-medium.bin"]
                            ))

                        if low_impact_words:
                            insights.append(PatternInsight(
                                insight_type="uk_vocabulary_weaknesses",
                                description=f"Identified {len(low_impact_words)} underperforming UK vocabulary items",
                                confidence=0.85,
                                recommendations={
                                    "action": "targeted_uk_vocabulary_training",
                                    "low_impact_words": [{"word": row['word_or_phrase'],
                                                        "impact": row['accuracy_impact']}
                                                       for row in low_impact_words[:10]],
                                    "training_priority": "immediate"
                                },
                                requires_retraining=True,
                                potential_accuracy_gain=0.05,
                                affected_models=["uk-english-fine-tuned-small.bin", "uk-english-fine-tuned-medium.bin"]
                            ))

        except Exception as e:
            logger.error(f"Error analyzing UK patterns: {e}")

        return insights

    async def _analyze_performance_patterns(self) -> List[PatternInsight]:
        """Analyze model performance patterns"""
        insights = []

        try:
            async with self.db_pool.acquire() as conn:
                # Get model performance trends
                performance_data = await conn.fetch("""
                    SELECT
                        model_name,
                        AVG(accuracy) as avg_accuracy,
                        COUNT(*) as sample_count,
                        STDDEV(accuracy) as accuracy_stddev,
                        AVG(processing_time_ms) as avg_processing_time,
                        AVG(CASE WHEN is_uk_english THEN accuracy END) as uk_accuracy
                    FROM model_performance
                    WHERE recorded_at > NOW() - INTERVAL '24 hours'
                    GROUP BY model_name
                    HAVING COUNT(*) > 10
                    ORDER BY avg_accuracy DESC
                """)

                if len(performance_data) >= 2:
                    # Find best and worst performing models
                    best_model = performance_data[0]
                    worst_model = performance_data[-1]

                    performance_gap = best_model['avg_accuracy'] - worst_model['avg_accuracy']

                    if performance_gap > 0.1:  # Significant performance difference
                        insights.append(PatternInsight(
                            insight_type="model_performance_disparity",
                            description=f"Large performance gap between models: {best_model['model_name']} "
                                      f"({best_model['avg_accuracy']:.3f}) vs {worst_model['model_name']} "
                                      f"({worst_model['avg_accuracy']:.3f})",
                            confidence=0.9,
                            recommendations={
                                "action": "model_weight_adjustment",
                                "increase_weight": best_model['model_name'],
                                "decrease_weight": worst_model['model_name'],
                                "performance_gap": float(performance_gap)
                            },
                            potential_accuracy_gain=min(0.08, performance_gap * 0.3),
                            affected_models=[best_model['model_name'], worst_model['model_name']]
                        ))

                    # Analyze processing time vs accuracy trade-offs
                    for model_data in performance_data:
                        accuracy = model_data['avg_accuracy']
                        processing_time = model_data['avg_processing_time']

                        # Flag slow models with marginal accuracy gains
                        if processing_time > 1000 and accuracy < 0.85:  # Slow and inaccurate
                            insights.append(PatternInsight(
                                insight_type="inefficient_model",
                                description=f"Model {model_data['model_name']} is slow ({processing_time:.0f}ms) "
                                          f"with low accuracy ({accuracy:.3f})",
                                confidence=0.8,
                                recommendations={
                                    "action": "reduce_model_usage",
                                    "model_name": model_data['model_name'],
                                    "alternative_action": "optimize_or_replace"
                                },
                                potential_accuracy_gain=0.02,
                                affected_models=[model_data['model_name']]
                            ))

                        # Flag high-variance models
                        if model_data['accuracy_stddev'] and model_data['accuracy_stddev'] > 0.2:
                            insights.append(PatternInsight(
                                insight_type="inconsistent_model",
                                description=f"Model {model_data['model_name']} shows high variance "
                                          f"(stddev: {model_data['accuracy_stddev']:.3f})",
                                confidence=0.75,
                                recommendations={
                                    "action": "model_stabilization",
                                    "model_name": model_data['model_name'],
                                    "variance": float(model_data['accuracy_stddev'])
                                },
                                requires_retraining=True,
                                potential_accuracy_gain=0.04,
                                affected_models=[model_data['model_name']]
                            ))

        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {e}")

        return insights

    async def _analyze_ensemble_patterns(self) -> List[PatternInsight]:
        """Analyze ensemble effectiveness patterns"""
        insights = []

        try:
            async with self.db_pool.acquire() as conn:
                # Analyze ensemble agreement patterns
                ensemble_data = await conn.fetch("""
                    SELECT
                        JSON_EXTRACT_PATH_TEXT(model_outputs, 'ensemble_agreement') as agreement,
                        confidence,
                        is_uk_english
                    FROM recognition_history
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    AND model_outputs::text LIKE '%ensemble_agreement%'
                    AND JSON_EXTRACT_PATH_TEXT(model_outputs, 'ensemble_agreement') != ''
                    LIMIT 1000
                """)

                if len(ensemble_data) > 20:
                    agreements = []
                    confidences = []
                    uk_flags = []

                    for row in ensemble_data:
                        try:
                            agreement = float(row['agreement'])
                            agreements.append(agreement)
                            confidences.append(row['confidence'])
                            uk_flags.append(row['is_uk_english'])
                        except (ValueError, TypeError):
                            continue

                    if agreements:
                        avg_agreement = np.mean(agreements)
                        avg_confidence = np.mean(confidences)

                        # Low ensemble agreement indicates potential for improvement
                        if avg_agreement < 0.7:
                            insights.append(PatternInsight(
                                insight_type="low_ensemble_agreement",
                                description=f"Low ensemble agreement ({avg_agreement:.3f}) suggests "
                                          f"models are inconsistent",
                                confidence=0.8,
                                recommendations={
                                    "action": "ensemble_optimization",
                                    "current_agreement": float(avg_agreement),
                                    "target_agreement": 0.85,
                                    "suggested_method": "model_weight_calibration"
                                },
                                potential_accuracy_gain=min(0.06, (0.85 - avg_agreement) * 0.1),
                                affected_models=["ensemble"]
                            ))

                        # High agreement with low confidence suggests systematic issues
                        high_agreement_low_conf = [
                            (a, c) for a, c in zip(agreements, confidences)
                            if a > 0.9 and c < 0.7
                        ]

                        if len(high_agreement_low_conf) > len(agreements) * 0.1:  # More than 10%
                            insights.append(PatternInsight(
                                insight_type="systematic_ensemble_issue",
                                description=f"High model agreement ({len(high_agreement_low_conf)} cases) "
                                          f"but low confidence suggests systematic recognition issues",
                                confidence=0.85,
                                recommendations={
                                    "action": "training_data_audit",
                                    "issue_cases": len(high_agreement_low_conf),
                                    "investigation_priority": "high"
                                },
                                requires_retraining=True,
                                potential_accuracy_gain=0.08,
                                affected_models=["all"]
                            ))

                        # UK English ensemble performance
                        uk_agreements = [a for a, uk in zip(agreements, uk_flags) if uk]
                        non_uk_agreements = [a for a, uk in zip(agreements, uk_flags) if not uk]

                        if uk_agreements and non_uk_agreements:
                            uk_avg_agreement = np.mean(uk_agreements)
                            non_uk_avg_agreement = np.mean(non_uk_agreements)

                            if non_uk_avg_agreement - uk_avg_agreement > 0.1:
                                insights.append(PatternInsight(
                                    insight_type="uk_ensemble_underperformance",
                                    description=f"UK English ensemble agreement lower than general "
                                              f"({uk_avg_agreement:.3f} vs {non_uk_avg_agreement:.3f})",
                                    confidence=0.9,
                                    recommendations={
                                        "action": "uk_ensemble_optimization",
                                        "agreement_gap": float(non_uk_avg_agreement - uk_avg_agreement),
                                        "uk_model_weight_increase": 0.1
                                    },
                                    potential_accuracy_gain=0.05,
                                    affected_models=["uk-english-fine-tuned-small.bin", "uk-english-fine-tuned-medium.bin"]
                                ))

        except Exception as e:
            logger.error(f"Error analyzing ensemble patterns: {e}")

        return insights

    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()