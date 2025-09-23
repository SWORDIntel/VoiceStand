"""
Model Optimizer for VoiceStand Learning System
Dynamic model selection and ensemble optimization
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from dataclasses import dataclass
import asyncpg
from concurrent.futures import ThreadPoolExecutor
import torch
import threading
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    model_name: str
    accuracy: float
    uk_accuracy: Optional[float]
    processing_time_ms: float
    memory_usage_mb: float
    sample_count: int
    last_updated: datetime
    stability_score: float  # Variance in performance

@dataclass
class EnsembleConfiguration:
    models: List[str]
    weights: Dict[str, float]
    expected_accuracy: float
    processing_time_ms: float
    confidence_threshold: float
    uk_optimized: bool

class ModelOptimizer:
    """Advanced model optimization and ensemble coordination"""

    def __init__(self, db_url: str, models_dir: str = "/app/models"):
        self.db_url = db_url
        self.models_dir = Path(models_dir)
        self.db_pool = None

        # Available models and their characteristics
        self.available_models = {
            "ggml-small.bin": {
                "size_mb": 244,
                "baseline_accuracy": 0.79,
                "avg_processing_time_ms": 150,
                "specialization": "general",
                "memory_efficient": True
            },
            "ggml-medium.bin": {
                "size_mb": 769,
                "baseline_accuracy": 0.85,
                "avg_processing_time_ms": 300,
                "specialization": "general",
                "memory_efficient": False
            },
            "ggml-large.bin": {
                "size_mb": 1550,
                "baseline_accuracy": 0.88,
                "avg_processing_time_ms": 600,
                "specialization": "general",
                "memory_efficient": False
            },
            "uk-english-fine-tuned-small.bin": {
                "size_mb": 260,
                "baseline_accuracy": 0.82,
                "avg_processing_time_ms": 180,
                "specialization": "uk_english",
                "memory_efficient": True
            },
            "uk-english-fine-tuned-medium.bin": {
                "size_mb": 785,
                "baseline_accuracy": 0.87,
                "avg_processing_time_ms": 350,
                "specialization": "uk_english",
                "memory_efficient": False
            }
        }

        # Current ensemble configuration
        self.current_ensemble = EnsembleConfiguration(
            models=["ggml-medium.bin", "ggml-large.bin", "uk-english-fine-tuned-medium.bin"],
            weights={"ggml-medium.bin": 0.3, "ggml-large.bin": 0.4, "uk-english-fine-tuned-medium.bin": 0.3},
            expected_accuracy=0.89,
            processing_time_ms=400,
            confidence_threshold=0.85,
            uk_optimized=True
        )

        # Performance tracking
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.performance_history = deque(maxlen=1000)
        self.optimization_lock = threading.Lock()

        # Optimization parameters
        self.accuracy_target = 0.95
        self.max_ensemble_size = 5
        self.min_ensemble_size = 2
        self.optimization_interval = 300  # 5 minutes

        # Learning rates for weight adjustment
        self.weight_learning_rate = 0.01
        self.performance_decay_factor = 0.95

    async def initialize(self):
        """Initialize model optimizer with database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=8,
                command_timeout=60
            )

            await self._load_model_performance_history()
            await self._initialize_ensemble_configurations()

            logger.info("Model optimizer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize model optimizer: {e}")
            return False

    async def optimize_ensemble(self):
        """Main optimization routine for ensemble configuration"""
        try:
            with self.optimization_lock:
                logger.info("Starting ensemble optimization...")

                # Get current performance metrics
                current_metrics = await self._collect_current_metrics()

                # Analyze performance patterns
                performance_insights = await self._analyze_performance_patterns(current_metrics)

                # Generate new ensemble configurations
                candidate_ensembles = await self._generate_candidate_ensembles(current_metrics, performance_insights)

                # Evaluate and select best ensemble
                best_ensemble = await self._evaluate_ensemble_candidates(candidate_ensembles)

                # Update ensemble configuration if better
                if await self._should_update_ensemble(best_ensemble):
                    await self._update_ensemble_configuration(best_ensemble)
                    logger.info(f"Updated ensemble configuration: {best_ensemble.models}")

                # Store optimization results
                await self._store_optimization_results(best_ensemble, performance_insights)

                logger.info("Ensemble optimization completed")

        except Exception as e:
            logger.error(f"Error during ensemble optimization: {e}")

    async def get_optimal_ensemble_for_context(self, context: Dict[str, Any]) -> EnsembleConfiguration:
        """Get optimal ensemble configuration for specific context"""
        try:
            is_uk_english = context.get("is_uk_english", False)
            noise_level = context.get("noise_level", 0.0)
            processing_time_constraint = context.get("max_processing_time_ms", 1000)
            accuracy_requirement = context.get("min_accuracy", 0.85)

            # Start with current ensemble
            base_ensemble = self.current_ensemble

            # Adjust for UK English
            if is_uk_english:
                uk_ensemble = await self._create_uk_optimized_ensemble(base_ensemble)
                if uk_ensemble.expected_accuracy > base_ensemble.expected_accuracy:
                    base_ensemble = uk_ensemble

            # Adjust for noise conditions
            if noise_level > 0.5:
                noise_robust_ensemble = await self._create_noise_robust_ensemble(base_ensemble)
                if noise_robust_ensemble.expected_accuracy > base_ensemble.expected_accuracy:
                    base_ensemble = noise_robust_ensemble

            # Adjust for processing time constraints
            if processing_time_constraint < base_ensemble.processing_time_ms:
                fast_ensemble = await self._create_fast_ensemble(base_ensemble, processing_time_constraint)
                base_ensemble = fast_ensemble

            # Ensure accuracy requirement is met
            if base_ensemble.expected_accuracy < accuracy_requirement:
                high_accuracy_ensemble = await self._create_high_accuracy_ensemble(accuracy_requirement)
                base_ensemble = high_accuracy_ensemble

            return base_ensemble

        except Exception as e:
            logger.error(f"Error getting optimal ensemble for context: {e}")
            return self.current_ensemble

    async def retrain_models(self, model_names: List[str]):
        """Trigger retraining for specified models"""
        try:
            for model_name in model_names:
                if model_name in self.available_models:
                    logger.info(f"Starting retraining for model: {model_name}")

                    # Generate training recommendations
                    training_config = await self._generate_training_config(model_name)

                    # Store retraining request
                    await self._store_retraining_request(model_name, training_config)

                    logger.info(f"Retraining request stored for {model_name}")

        except Exception as e:
            logger.error(f"Error triggering model retraining: {e}")

    async def get_loaded_models_count(self) -> int:
        """Get count of currently loaded models"""
        return len(self.current_ensemble.models)

    async def _collect_current_metrics(self) -> Dict[str, ModelPerformanceMetrics]:
        """Collect current performance metrics for all models"""
        metrics = {}

        try:
            async with self.db_pool.acquire() as conn:
                # Get recent performance data for each model
                for model_name in self.available_models.keys():
                    model_stats = await conn.fetchrow("""
                        SELECT
                            AVG(accuracy) as avg_accuracy,
                            AVG(CASE WHEN is_uk_english THEN accuracy END) as uk_accuracy,
                            AVG(processing_time_ms) as avg_processing_time,
                            COUNT(*) as sample_count,
                            STDDEV(accuracy) as accuracy_stddev
                        FROM model_performance
                        WHERE model_name = $1
                        AND recorded_at > NOW() - INTERVAL '24 hours'
                    """, model_name)

                    if model_stats and model_stats['sample_count'] > 0:
                        metrics[model_name] = ModelPerformanceMetrics(
                            model_name=model_name,
                            accuracy=float(model_stats['avg_accuracy']),
                            uk_accuracy=float(model_stats['uk_accuracy']) if model_stats['uk_accuracy'] else None,
                            processing_time_ms=float(model_stats['avg_processing_time']),
                            memory_usage_mb=self.available_models[model_name]["size_mb"],
                            sample_count=model_stats['sample_count'],
                            last_updated=datetime.now(),
                            stability_score=1.0 - min(1.0, float(model_stats['accuracy_stddev'] or 0.0) * 5)
                        )
                    else:
                        # Use baseline metrics if no recent data
                        baseline = self.available_models[model_name]
                        metrics[model_name] = ModelPerformanceMetrics(
                            model_name=model_name,
                            accuracy=baseline["baseline_accuracy"],
                            uk_accuracy=None,
                            processing_time_ms=baseline["avg_processing_time_ms"],
                            memory_usage_mb=baseline["size_mb"],
                            sample_count=0,
                            last_updated=datetime.now(),
                            stability_score=0.8  # Default for untested models
                        )

            self.model_metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Error collecting current metrics: {e}")
            return {}

    async def _analyze_performance_patterns(self, metrics: Dict[str, ModelPerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance patterns to identify optimization opportunities"""
        insights = {}

        try:
            # Identify best and worst performing models
            sorted_models = sorted(metrics.items(), key=lambda x: x[1].accuracy, reverse=True)

            if len(sorted_models) >= 2:
                best_model = sorted_models[0]
                worst_model = sorted_models[-1]

                insights["performance_leader"] = {
                    "model": best_model[0],
                    "accuracy": best_model[1].accuracy,
                    "uk_accuracy": best_model[1].uk_accuracy
                }

                insights["performance_laggard"] = {
                    "model": worst_model[0],
                    "accuracy": worst_model[1].accuracy,
                    "improvement_needed": best_model[1].accuracy - worst_model[1].accuracy
                }

            # Analyze UK English performance
            uk_performers = {}
            for model_name, metric in metrics.items():
                if metric.uk_accuracy is not None:
                    uk_performers[model_name] = metric.uk_accuracy

            if uk_performers:
                best_uk_model = max(uk_performers.items(), key=lambda x: x[1])
                insights["uk_performance"] = {
                    "best_uk_model": best_uk_model[0],
                    "uk_accuracy": best_uk_model[1],
                    "general_vs_uk_gap": {}
                }

                # Calculate UK vs general performance gaps
                for model_name in uk_performers:
                    general_acc = metrics[model_name].accuracy
                    uk_acc = uk_performers[model_name]
                    insights["uk_performance"]["general_vs_uk_gap"][model_name] = general_acc - uk_acc

            # Analyze processing efficiency
            efficiency_scores = {}
            for model_name, metric in metrics.items():
                # Efficiency = accuracy / (processing_time * memory_usage)
                efficiency = metric.accuracy / (metric.processing_time_ms * metric.memory_usage_mb / 1000)
                efficiency_scores[model_name] = efficiency

            most_efficient = max(efficiency_scores.items(), key=lambda x: x[1])
            insights["efficiency"] = {
                "most_efficient_model": most_efficient[0],
                "efficiency_score": most_efficient[1],
                "efficiency_ranking": sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
            }

            # Identify models needing improvement
            insights["improvement_candidates"] = []
            for model_name, metric in metrics.items():
                if metric.accuracy < 0.85 or metric.stability_score < 0.7:
                    insights["improvement_candidates"].append({
                        "model": model_name,
                        "accuracy": metric.accuracy,
                        "stability": metric.stability_score,
                        "issues": []
                    })

                    if metric.accuracy < 0.85:
                        insights["improvement_candidates"][-1]["issues"].append("low_accuracy")
                    if metric.stability_score < 0.7:
                        insights["improvement_candidates"][-1]["issues"].append("high_variance")

            return insights

        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {e}")
            return {}

    async def _generate_candidate_ensembles(self, metrics: Dict[str, ModelPerformanceMetrics],
                                          insights: Dict[str, Any]) -> List[EnsembleConfiguration]:
        """Generate candidate ensemble configurations"""
        candidates = []

        try:
            # Current ensemble (baseline)
            candidates.append(self.current_ensemble)

            # High accuracy ensemble (top performing models)
            sorted_by_accuracy = sorted(metrics.items(), key=lambda x: x[1].accuracy, reverse=True)
            top_models = [model for model, _ in sorted_by_accuracy[:self.max_ensemble_size]]

            if len(top_models) >= self.min_ensemble_size:
                accuracy_weights = self._calculate_accuracy_weights([metrics[m] for m in top_models])
                candidates.append(EnsembleConfiguration(
                    models=top_models,
                    weights=accuracy_weights,
                    expected_accuracy=np.average([metrics[m].accuracy for m in top_models],
                                               weights=list(accuracy_weights.values())),
                    processing_time_ms=np.average([metrics[m].processing_time_ms for m in top_models]),
                    confidence_threshold=0.85,
                    uk_optimized=any("uk-english" in m for m in top_models)
                ))

            # UK-optimized ensemble
            uk_models = [m for m in metrics.keys() if "uk-english" in m or metrics[m].uk_accuracy is not None]
            if len(uk_models) >= 2:
                # Add best general model
                general_models = [m for m in metrics.keys() if m not in uk_models]
                if general_models:
                    best_general = max(general_models, key=lambda x: metrics[x].accuracy)
                    uk_models.insert(0, best_general)

                uk_weights = self._calculate_uk_optimized_weights([metrics[m] for m in uk_models[:self.max_ensemble_size]])
                candidates.append(EnsembleConfiguration(
                    models=uk_models[:self.max_ensemble_size],
                    weights=uk_weights,
                    expected_accuracy=self._estimate_uk_ensemble_accuracy(uk_models[:self.max_ensemble_size], metrics),
                    processing_time_ms=np.average([metrics[m].processing_time_ms for m in uk_models[:self.max_ensemble_size]]),
                    confidence_threshold=0.8,
                    uk_optimized=True
                ))

            # Efficient ensemble (balance accuracy and speed)
            efficiency_scores = {}
            for model_name, metric in metrics.items():
                efficiency = metric.accuracy / (metric.processing_time_ms / 100)  # Normalized efficiency
                efficiency_scores[model_name] = efficiency

            efficient_models = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
            efficient_model_names = [model for model, _ in efficient_models[:3]]

            if len(efficient_model_names) >= 2:
                efficient_weights = self._calculate_efficiency_weights([metrics[m] for m in efficient_model_names])
                candidates.append(EnsembleConfiguration(
                    models=efficient_model_names,
                    weights=efficient_weights,
                    expected_accuracy=np.average([metrics[m].accuracy for m in efficient_model_names],
                                               weights=list(efficient_weights.values())),
                    processing_time_ms=np.average([metrics[m].processing_time_ms for m in efficient_model_names]),
                    confidence_threshold=0.8,
                    uk_optimized=any("uk-english" in m for m in efficient_model_names)
                ))

            # Specialized ensemble based on insights
            if "performance_leader" in insights:
                leader = insights["performance_leader"]["model"]
                # Create ensemble around the best performing model
                supporting_models = [m for m in metrics.keys() if m != leader and metrics[m].accuracy > 0.8]

                if supporting_models:
                    specialized_models = [leader] + supporting_models[:2]
                    specialized_weights = {leader: 0.6}  # Higher weight for leader
                    remaining_weight = 0.4 / len(supporting_models[:2])
                    for model in supporting_models[:2]:
                        specialized_weights[model] = remaining_weight

                    candidates.append(EnsembleConfiguration(
                        models=specialized_models,
                        weights=specialized_weights,
                        expected_accuracy=metrics[leader].accuracy * 0.6 +
                                        np.mean([metrics[m].accuracy for m in supporting_models[:2]]) * 0.4,
                        processing_time_ms=metrics[leader].processing_time_ms * 0.6 +
                                         np.mean([metrics[m].processing_time_ms for m in supporting_models[:2]]) * 0.4,
                        confidence_threshold=0.85,
                        uk_optimized=any("uk-english" in m for m in specialized_models)
                    ))

            return candidates

        except Exception as e:
            logger.error(f"Error generating candidate ensembles: {e}")
            return [self.current_ensemble]

    def _calculate_accuracy_weights(self, model_metrics: List[ModelPerformanceMetrics]) -> Dict[str, float]:
        """Calculate weights based on model accuracy"""
        accuracies = [m.accuracy for m in model_metrics]
        total_accuracy = sum(accuracies)

        if total_accuracy == 0:
            # Equal weights if no accuracy data
            weight = 1.0 / len(model_metrics)
            return {m.model_name: weight for m in model_metrics}

        return {m.model_name: m.accuracy / total_accuracy for m in model_metrics}

    def _calculate_uk_optimized_weights(self, model_metrics: List[ModelPerformanceMetrics]) -> Dict[str, float]:
        """Calculate weights optimized for UK English performance"""
        weights = {}
        total_weight = 0.0

        for metric in model_metrics:
            if "uk-english" in metric.model_name:
                # Higher base weight for UK-specialized models
                base_weight = 0.4
            else:
                base_weight = 0.2

            # Adjust by UK accuracy if available
            if metric.uk_accuracy is not None:
                weight = base_weight * metric.uk_accuracy
            else:
                weight = base_weight * metric.accuracy * 0.8  # Penalty for no UK data

            weights[metric.model_name] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight

        return weights

    def _calculate_efficiency_weights(self, model_metrics: List[ModelPerformanceMetrics]) -> Dict[str, float]:
        """Calculate weights based on efficiency (accuracy per processing time)"""
        efficiencies = []
        for metric in model_metrics:
            efficiency = metric.accuracy / (metric.processing_time_ms / 100)
            efficiencies.append(efficiency)

        total_efficiency = sum(efficiencies)

        if total_efficiency == 0:
            weight = 1.0 / len(model_metrics)
            return {m.model_name: weight for m in model_metrics}

        return {metric.model_name: efficiency / total_efficiency
                for metric, efficiency in zip(model_metrics, efficiencies)}

    def _estimate_uk_ensemble_accuracy(self, models: List[str], metrics: Dict[str, ModelPerformanceMetrics]) -> float:
        """Estimate ensemble accuracy for UK English"""
        uk_accuracies = []
        weights = []

        for model in models:
            metric = metrics[model]
            if metric.uk_accuracy is not None:
                uk_accuracies.append(metric.uk_accuracy)
                weights.append(1.0)
            else:
                # Use general accuracy with penalty for non-UK models
                uk_accuracies.append(metric.accuracy * 0.9)
                weights.append(0.8)

        if uk_accuracies:
            return np.average(uk_accuracies, weights=weights)
        else:
            return 0.85  # Default estimate

    async def _evaluate_ensemble_candidates(self, candidates: List[EnsembleConfiguration]) -> EnsembleConfiguration:
        """Evaluate ensemble candidates and select the best one"""
        try:
            best_ensemble = candidates[0]
            best_score = 0.0

            for ensemble in candidates:
                # Calculate composite score
                accuracy_score = ensemble.expected_accuracy * 40  # 40% weight on accuracy
                speed_score = max(0, (1000 - ensemble.processing_time_ms) / 1000) * 20  # 20% weight on speed
                uk_score = 20 if ensemble.uk_optimized else 0  # 20% weight on UK optimization

                # Stability bonus
                model_count_score = min(20, len(ensemble.models) * 4)  # Up to 20% for ensemble size

                total_score = accuracy_score + speed_score + uk_score + model_count_score

                if total_score > best_score:
                    best_score = total_score
                    best_ensemble = ensemble

            logger.info(f"Best ensemble score: {best_score:.2f} for models: {best_ensemble.models}")
            return best_ensemble

        except Exception as e:
            logger.error(f"Error evaluating ensemble candidates: {e}")
            return candidates[0]

    async def _should_update_ensemble(self, candidate: EnsembleConfiguration) -> bool:
        """Determine if ensemble should be updated"""
        # Update if significant accuracy improvement
        accuracy_improvement = candidate.expected_accuracy - self.current_ensemble.expected_accuracy

        # Update if processing time improvement without significant accuracy loss
        time_improvement = self.current_ensemble.processing_time_ms - candidate.processing_time_ms

        # Update thresholds
        min_accuracy_improvement = 0.02  # 2% accuracy improvement
        min_time_improvement = 100  # 100ms time improvement
        max_accuracy_loss = 0.01  # Don't lose more than 1% accuracy

        return (accuracy_improvement > min_accuracy_improvement or
                (time_improvement > min_time_improvement and accuracy_improvement > -max_accuracy_loss))

    async def _update_ensemble_configuration(self, new_ensemble: EnsembleConfiguration):
        """Update the current ensemble configuration"""
        try:
            self.current_ensemble = new_ensemble

            # Store updated configuration
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ensemble_performance
                    (model_combination, average_accuracy, uk_english_accuracy,
                     processing_time_avg_ms, sample_count)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                new_ensemble.models,
                new_ensemble.expected_accuracy,
                new_ensemble.expected_accuracy if new_ensemble.uk_optimized else None,
                int(new_ensemble.processing_time_ms),
                1)

            logger.info(f"Updated ensemble configuration: {new_ensemble.models}")

        except Exception as e:
            logger.error(f"Error updating ensemble configuration: {e}")

    async def _store_optimization_results(self, best_ensemble: EnsembleConfiguration, insights: Dict[str, Any]):
        """Store optimization results for analysis"""
        try:
            async with self.db_pool.acquire() as conn:
                optimization_data = {
                    "timestamp": datetime.now().isoformat(),
                    "best_ensemble": {
                        "models": best_ensemble.models,
                        "weights": best_ensemble.weights,
                        "expected_accuracy": best_ensemble.expected_accuracy,
                        "processing_time_ms": best_ensemble.processing_time_ms
                    },
                    "insights": insights
                }

                # Store in a simple table (you might want to create this table)
                await conn.execute("""
                    INSERT INTO model_performance
                    (model_name, accuracy, processing_time_ms, recorded_at, is_uk_english)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                "ensemble_optimization",
                best_ensemble.expected_accuracy,
                int(best_ensemble.processing_time_ms),
                datetime.now(),
                best_ensemble.uk_optimized)

        except Exception as e:
            logger.error(f"Error storing optimization results: {e}")

    async def _create_uk_optimized_ensemble(self, base_ensemble: EnsembleConfiguration) -> EnsembleConfiguration:
        """Create UK-optimized version of ensemble"""
        uk_models = [m for m in base_ensemble.models if "uk-english" in m]

        if not uk_models and self.model_metrics:
            # Add UK models if none present
            available_uk_models = [m for m in self.model_metrics.keys() if "uk-english" in m]
            if available_uk_models:
                best_uk_model = max(available_uk_models,
                                  key=lambda x: self.model_metrics[x].uk_accuracy or 0)
                uk_models = base_ensemble.models[:2] + [best_uk_model]

        if uk_models:
            uk_weights = {}
            total_weight = 0.0

            for model in uk_models:
                if "uk-english" in model:
                    weight = 0.4
                else:
                    weight = 0.3
                uk_weights[model] = weight
                total_weight += weight

            # Normalize
            for model in uk_weights:
                uk_weights[model] /= total_weight

            return EnsembleConfiguration(
                models=uk_models,
                weights=uk_weights,
                expected_accuracy=base_ensemble.expected_accuracy + 0.02,  # UK boost
                processing_time_ms=base_ensemble.processing_time_ms * 1.1,
                confidence_threshold=0.8,
                uk_optimized=True
            )

        return base_ensemble

    async def _create_noise_robust_ensemble(self, base_ensemble: EnsembleConfiguration) -> EnsembleConfiguration:
        """Create noise-robust version of ensemble"""
        # Prefer larger models for noise robustness
        noise_robust_models = [m for m in base_ensemble.models if "large" in m or "medium" in m]

        if len(noise_robust_models) < 2 and self.model_metrics:
            # Add robust models
            robust_candidates = [m for m in self.model_metrics.keys() if "large" in m]
            noise_robust_models.extend(robust_candidates[:3 - len(noise_robust_models)])

        if noise_robust_models:
            return EnsembleConfiguration(
                models=noise_robust_models[:3],
                weights={m: 1.0/len(noise_robust_models[:3]) for m in noise_robust_models[:3]},
                expected_accuracy=base_ensemble.expected_accuracy - 0.01,  # Slight penalty for noise focus
                processing_time_ms=base_ensemble.processing_time_ms * 1.2,
                confidence_threshold=0.7,  # Lower threshold for noisy conditions
                uk_optimized=base_ensemble.uk_optimized
            )

        return base_ensemble

    async def _create_fast_ensemble(self, base_ensemble: EnsembleConfiguration,
                                  time_constraint: float) -> EnsembleConfiguration:
        """Create fast ensemble within time constraints"""
        if self.model_metrics:
            # Sort by processing time
            fast_models = sorted(self.model_metrics.items(),
                               key=lambda x: x[1].processing_time_ms)

            selected_models = []
            cumulative_time = 0.0

            for model_name, metric in fast_models:
                if cumulative_time + metric.processing_time_ms <= time_constraint:
                    selected_models.append(model_name)
                    cumulative_time += metric.processing_time_ms

                    if len(selected_models) >= 3:  # Enough models
                        break

            if len(selected_models) >= 2:
                return EnsembleConfiguration(
                    models=selected_models,
                    weights={m: 1.0/len(selected_models) for m in selected_models},
                    expected_accuracy=base_ensemble.expected_accuracy - 0.03,  # Penalty for speed focus
                    processing_time_ms=cumulative_time / len(selected_models),
                    confidence_threshold=0.8,
                    uk_optimized=any("uk-english" in m for m in selected_models)
                )

        return base_ensemble

    async def _create_high_accuracy_ensemble(self, accuracy_requirement: float) -> EnsembleConfiguration:
        """Create high-accuracy ensemble"""
        if self.model_metrics:
            # Use all high-performing models
            high_accuracy_models = [
                name for name, metric in self.model_metrics.items()
                if metric.accuracy >= accuracy_requirement - 0.05
            ]

            if len(high_accuracy_models) >= 2:
                return EnsembleConfiguration(
                    models=high_accuracy_models[:self.max_ensemble_size],
                    weights={m: 1.0/len(high_accuracy_models[:self.max_ensemble_size])
                            for m in high_accuracy_models[:self.max_ensemble_size]},
                    expected_accuracy=accuracy_requirement,
                    processing_time_ms=800,  # Accept higher processing time
                    confidence_threshold=0.9,
                    uk_optimized=any("uk-english" in m for m in high_accuracy_models)
                )

        return self.current_ensemble

    async def _generate_training_config(self, model_name: str) -> Dict[str, Any]:
        """Generate training configuration for model retraining"""
        base_config = {
            "model_name": model_name,
            "training_type": "fine_tuning",
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 10,
            "validation_split": 0.2
        }

        # Customize based on model type
        if "uk-english" in model_name:
            base_config.update({
                "focus": "uk_dialect",
                "training_data_filter": "uk_english",
                "augmentation": "uk_accent_variations"
            })

        if "small" in model_name:
            base_config.update({
                "batch_size": 32,
                "learning_rate": 0.0005
            })

        return base_config

    async def _store_retraining_request(self, model_name: str, config: Dict[str, Any]):
        """Store retraining request in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO model_performance
                    (model_name, accuracy, processing_time_ms, recorded_at, is_uk_english)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                f"retraining_request_{model_name}",
                0.0,  # Placeholder
                0,    # Placeholder
                datetime.now(),
                "uk-english" in model_name)

                logger.info(f"Stored retraining request for {model_name}")

        except Exception as e:
            logger.error(f"Error storing retraining request: {e}")

    async def _load_model_performance_history(self):
        """Load historical model performance data"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT model_name, accuracy, processing_time_ms, recorded_at, is_uk_english
                    FROM model_performance
                    WHERE recorded_at > NOW() - INTERVAL '7 days'
                    ORDER BY recorded_at DESC
                    LIMIT 1000
                """)

                for row in rows:
                    self.performance_history.append({
                        "model_name": row["model_name"],
                        "accuracy": row["accuracy"],
                        "processing_time_ms": row["processing_time_ms"],
                        "recorded_at": row["recorded_at"],
                        "is_uk_english": row["is_uk_english"]
                    })

                logger.info(f"Loaded {len(self.performance_history)} performance records")

        except Exception as e:
            logger.error(f"Error loading performance history: {e}")

    async def _initialize_ensemble_configurations(self):
        """Initialize ensemble configurations from database"""
        try:
            async with self.db_pool.acquire() as conn:
                latest_ensemble = await conn.fetchrow("""
                    SELECT model_combination, average_accuracy, processing_time_avg_ms, uk_english_accuracy
                    FROM ensemble_performance
                    ORDER BY last_updated DESC
                    LIMIT 1
                """)

                if latest_ensemble:
                    self.current_ensemble = EnsembleConfiguration(
                        models=latest_ensemble["model_combination"],
                        weights={model: 1.0/len(latest_ensemble["model_combination"])
                                for model in latest_ensemble["model_combination"]},
                        expected_accuracy=latest_ensemble["average_accuracy"],
                        processing_time_ms=latest_ensemble["processing_time_avg_ms"],
                        confidence_threshold=0.85,
                        uk_optimized=latest_ensemble["uk_english_accuracy"] is not None
                    )

                    logger.info(f"Loaded ensemble configuration: {self.current_ensemble.models}")

        except Exception as e:
            logger.error(f"Error initializing ensemble configurations: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()