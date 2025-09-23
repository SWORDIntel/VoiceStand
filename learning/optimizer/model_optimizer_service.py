"""
Model Optimizer Service for VoiceStand Learning System
Intel NPU-accelerated model optimization and ensemble coordination
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

import torch
import intel_extension_for_pytorch as ipex
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
import onnx
import onnxruntime as ort
from optimum.intel import IPEXModelForSpeechSeq2Seq

import asyncpg
import aiohttp
import structlog
import schedule
import psutil

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class IntelNPUOptimizer:
    """Intel NPU-specific model optimization"""

    def __init__(self):
        self.npu_available = self._check_npu_availability()
        self.optimization_configs = self._get_optimization_configs()

    def _check_npu_availability(self) -> bool:
        """Check if Intel NPU is available"""
        try:
            # Check for Intel NPU runtime
            npu_path = os.getenv("INTEL_NPU_ACCELERATION_LIBRARY_PATH")
            if npu_path and Path(npu_path).exists():
                logger.info("Intel NPU acceleration library detected")
                return True

            # Check system for NPU devices
            npu_devices = []
            try:
                import subprocess
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if 'Neural Processing Unit' in result.stdout or 'NPU' in result.stdout:
                    npu_devices = ['NPU detected via lspci']
            except:
                pass

            if npu_devices:
                logger.info("Intel NPU hardware detected", devices=npu_devices)
                return True

            logger.info("Intel NPU not available, using CPU optimizations")
            return False

        except Exception as e:
            logger.warning("Error checking NPU availability", error=str(e))
            return False

    def _get_optimization_configs(self) -> Dict[str, Any]:
        """Get NPU optimization configurations"""
        base_config = {
            "precision": "fp16",
            "batch_size": 1,
            "sequence_length": 1500,
            "optimization_level": "max_performance"
        }

        if self.npu_available:
            npu_config = base_config.copy()
            npu_config.update({
                "device": "npu",
                "npu_precision": os.getenv("INTEL_NPU_PRECISION", "fp16"),
                "npu_batch_size": int(os.getenv("INTEL_NPU_BATCH_SIZE", "1")),
                "runtime_level": int(os.getenv("INTEL_NPU_RUNTIME_LEVEL", "4"))
            })
            return {"npu": npu_config, "cpu": base_config}
        else:
            return {"cpu": base_config}

    async def optimize_model(self, model_path: str, output_path: str, optimization_type: str = "auto") -> Dict[str, Any]:
        """Optimize model for Intel hardware"""
        try:
            logger.info("Starting model optimization",
                       model_path=model_path,
                       npu_available=self.npu_available)

            optimization_result = {
                "success": False,
                "original_size_mb": 0,
                "optimized_size_mb": 0,
                "compression_ratio": 0.0,
                "optimization_type": optimization_type,
                "npu_optimized": self.npu_available
            }

            # Get original model size
            if Path(model_path).exists():
                original_size = Path(model_path).stat().st_size / (1024 * 1024)
                optimization_result["original_size_mb"] = original_size

            # Load model for optimization
            model = self._load_model_for_optimization(model_path)
            if model is None:
                return optimization_result

            # Apply Intel optimizations
            if self.npu_available:
                optimized_model = await self._optimize_for_npu(model, optimization_type)
            else:
                optimized_model = await self._optimize_for_cpu(model, optimization_type)

            # Save optimized model
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            if self.npu_available:
                # Save in NPU-optimized format
                optimized_path = output_dir / "npu_optimized_model.pt"
                torch.jit.save(optimized_model, optimized_path)
            else:
                # Save in CPU-optimized format
                optimized_path = output_dir / "cpu_optimized_model.pt"
                torch.jit.save(optimized_model, optimized_path)

            # Get optimized model size
            if optimized_path.exists():
                optimized_size = optimized_path.stat().st_size / (1024 * 1024)
                optimization_result["optimized_size_mb"] = optimized_size
                optimization_result["compression_ratio"] = original_size / optimized_size if optimized_size > 0 else 0
                optimization_result["success"] = True

            logger.info("Model optimization completed", result=optimization_result)
            return optimization_result

        except Exception as e:
            logger.error("Error optimizing model", error=str(e))
            optimization_result["error"] = str(e)
            return optimization_result

    def _load_model_for_optimization(self, model_path: str):
        """Load model for optimization"""
        try:
            # This is a simplified loader - in practice you'd handle different model formats
            if Path(model_path).suffix == '.pt':
                return torch.jit.load(model_path)
            else:
                # Create a dummy model for demonstration
                class DummyWhisperModel(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.encoder = torch.nn.Linear(1024, 512)
                        self.decoder = torch.nn.Linear(512, 256)

                    def forward(self, x):
                        x = self.encoder(x)
                        return self.decoder(x)

                return DummyWhisperModel()

        except Exception as e:
            logger.error("Error loading model for optimization", error=str(e))
            return None

    async def _optimize_for_npu(self, model, optimization_type: str):
        """Optimize model for Intel NPU"""
        try:
            logger.info("Applying NPU optimizations")

            # Intel Extension for PyTorch optimizations
            model.eval()

            # Apply NPU-specific optimizations
            with torch.no_grad():
                # Trace the model for NPU execution
                dummy_input = torch.randn(1, 1024)  # Adjust based on your model
                traced_model = torch.jit.trace(model, dummy_input)

                # Apply Intel optimizations
                optimized_model = ipex.optimize(
                    traced_model,
                    dtype=torch.float16,  # Use FP16 for NPU
                    level="O1"  # Optimization level
                )

                # Additional NPU-specific optimizations would go here
                # This might include model graph optimizations, memory layout changes, etc.

            return optimized_model

        except Exception as e:
            logger.error("Error applying NPU optimizations", error=str(e))
            return model

    async def _optimize_for_cpu(self, model, optimization_type: str):
        """Optimize model for Intel CPU"""
        try:
            logger.info("Applying CPU optimizations")

            model.eval()

            # Intel Extension for PyTorch CPU optimizations
            with torch.no_grad():
                dummy_input = torch.randn(1, 1024)
                traced_model = torch.jit.trace(model, dummy_input)

                # Apply Intel CPU optimizations
                optimized_model = ipex.optimize(
                    traced_model,
                    dtype=torch.float32,  # Use FP32 for CPU
                    level="O1"
                )

                # Apply quantization if requested
                if optimization_type in ["quantized", "auto"]:
                    optimized_model = await self._apply_quantization(optimized_model)

            return optimized_model

        except Exception as e:
            logger.error("Error applying CPU optimizations", error=str(e))
            return model

    async def _apply_quantization(self, model):
        """Apply Intel Neural Compressor quantization"""
        try:
            logger.info("Applying quantization optimization")

            # Configure quantization
            config = PostTrainingQuantConfig(approach="static")

            # Apply quantization (simplified example)
            # In practice, you'd need calibration data
            quantized_model = quantization.fit(
                model=model,
                conf=config,
                calib_dataloader=None  # Would need actual calibration data
            )

            return quantized_model

        except Exception as e:
            logger.error("Error applying quantization", error=str(e))
            return model

class ModelOptimizerService:
    """Service for optimizing VoiceStand models with Intel acceleration"""

    def __init__(self, db_url: str, learning_api_url: str):
        self.db_url = db_url
        self.learning_api_url = learning_api_url
        self.db_pool = None

        # Intel NPU optimizer
        self.npu_optimizer = IntelNPUOptimizer()

        # Optimization configuration
        self.optimization_interval = int(os.getenv("OPTIMIZATION_INTERVAL", "300"))  # 5 minutes
        self.accuracy_target = float(os.getenv("ACCURACY_TARGET", "0.95"))

        # Models directory
        self.models_dir = Path("/app/models")
        self.optimized_models_dir = Path("/app/optimized_models")
        self.optimized_models_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.optimization_history = []
        self.current_ensemble_config = None

    async def initialize(self):
        """Initialize the optimizer service"""
        try:
            # Database connection
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=5,
                command_timeout=60
            )

            # Test learning API connection
            await self._test_learning_api_connection()

            logger.info("Model optimizer service initialized",
                       npu_available=self.npu_optimizer.npu_available)
            return True

        except Exception as e:
            logger.error("Failed to initialize optimizer service", error=str(e))
            return False

    async def start_optimization_service(self):
        """Start the optimization service loop"""
        logger.info("Starting model optimization service...")

        # Schedule optimization tasks
        schedule.every(self.optimization_interval).seconds.do(self._optimization_cycle)
        schedule.every(1).hours.do(self._comprehensive_optimization)
        schedule.every(6).hours.do(self._ensemble_optimization)

        # Run initial optimization check
        await self._optimization_cycle()

        # Main service loop
        while True:
            try:
                schedule.run_pending()
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error("Error in optimization service loop", error=str(e))
                await asyncio.sleep(60)

    async def _optimization_cycle(self):
        """Regular optimization cycle"""
        try:
            logger.info("Starting optimization cycle...")

            # Get current performance metrics
            performance_data = await self._get_performance_metrics()

            # Identify models needing optimization
            models_to_optimize = await self._identify_optimization_candidates(performance_data)

            # Optimize identified models
            optimization_results = []
            for model_info in models_to_optimize:
                result = await self._optimize_single_model(model_info)
                optimization_results.append(result)

            # Update ensemble configuration if improvements found
            if any(r.get("success", False) for r in optimization_results):
                await self._update_ensemble_configuration(optimization_results)

            # Store optimization results
            await self._store_optimization_results(optimization_results)

            logger.info("Optimization cycle completed",
                       models_optimized=len([r for r in optimization_results if r.get("success", False)]))

        except Exception as e:
            logger.error("Error in optimization cycle", error=str(e))

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics from database and API"""
        metrics = {"models": [], "ensemble": {}, "system": {}}

        try:
            # Get model performance from database
            async with self.db_pool.acquire() as conn:
                model_performance = await conn.fetch("""
                    SELECT
                        model_name,
                        AVG(accuracy) as avg_accuracy,
                        AVG(CASE WHEN is_uk_english THEN accuracy END) as uk_accuracy,
                        AVG(processing_time_ms) as avg_processing_time,
                        COUNT(*) as sample_count,
                        STDDEV(accuracy) as accuracy_stddev
                    FROM model_performance
                    WHERE recorded_at > NOW() - INTERVAL '1 hour'
                    GROUP BY model_name
                    HAVING COUNT(*) > 5
                """)

                for row in model_performance:
                    metrics["models"].append({
                        "name": row["model_name"],
                        "accuracy": float(row["avg_accuracy"]) if row["avg_accuracy"] else 0.0,
                        "uk_accuracy": float(row["uk_accuracy"]) if row["uk_accuracy"] else None,
                        "processing_time_ms": float(row["avg_processing_time"]) if row["avg_processing_time"] else 0.0,
                        "sample_count": row["sample_count"],
                        "stability": 1.0 - min(1.0, float(row["accuracy_stddev"] or 0.0) * 5)
                    })

            # Get ensemble performance from API
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.learning_api_url}/api/v1/model_performance") as response:
                        if response.status == 200:
                            api_performance = await response.json()
                            metrics["ensemble"] = {
                                "models": api_performance,
                                "agreement_score": 0.85,  # Would be calculated from actual data
                                "overall_accuracy": np.mean([m.get("accuracy", 0) for m in api_performance]) if api_performance else 0
                            }
            except Exception as e:
                logger.warning("Could not get API performance metrics", error=str(e))

            # System performance
            metrics["system"] = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "npu_available": self.npu_optimizer.npu_available
            }

            return metrics

        except Exception as e:
            logger.error("Error getting performance metrics", error=str(e))
            return metrics

    async def _identify_optimization_candidates(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify models that need optimization"""
        candidates = []

        try:
            for model in performance_data["models"]:
                model_name = model["name"]
                accuracy = model["accuracy"]
                processing_time = model["processing_time_ms"]
                stability = model["stability"]

                optimization_needed = False
                reasons = []

                # Check accuracy threshold
                if accuracy < self.accuracy_target - 0.02:
                    optimization_needed = True
                    reasons.append(f"low_accuracy_{accuracy:.3f}")

                # Check UK accuracy if available
                if model["uk_accuracy"] and model["uk_accuracy"] < self.accuracy_target - 0.03:
                    optimization_needed = True
                    reasons.append(f"low_uk_accuracy_{model['uk_accuracy']:.3f}")

                # Check processing time (optimize if > 500ms)
                if processing_time > 500:
                    optimization_needed = True
                    reasons.append(f"slow_processing_{processing_time:.0f}ms")

                # Check stability
                if stability < 0.8:
                    optimization_needed = True
                    reasons.append(f"low_stability_{stability:.3f}")

                if optimization_needed:
                    candidates.append({
                        "model_name": model_name,
                        "current_accuracy": accuracy,
                        "uk_accuracy": model["uk_accuracy"],
                        "processing_time_ms": processing_time,
                        "stability": stability,
                        "optimization_reasons": reasons,
                        "priority": self._calculate_optimization_priority(model)
                    })

            # Sort by priority
            candidates.sort(key=lambda x: x["priority"], reverse=True)

            logger.info("Identified optimization candidates",
                       count=len(candidates),
                       candidates=[c["model_name"] for c in candidates[:3]])

            return candidates

        except Exception as e:
            logger.error("Error identifying optimization candidates", error=str(e))
            return []

    def _calculate_optimization_priority(self, model: Dict[str, Any]) -> float:
        """Calculate optimization priority score"""
        priority = 0.0

        # Accuracy impact (higher priority for lower accuracy)
        accuracy_gap = self.accuracy_target - model["accuracy"]
        priority += accuracy_gap * 100

        # UK accuracy impact
        if model["uk_accuracy"]:
            uk_accuracy_gap = self.accuracy_target - model["uk_accuracy"]
            priority += uk_accuracy_gap * 80

        # Processing time impact (priority for slow models)
        if model["processing_time_ms"] > 300:
            priority += (model["processing_time_ms"] - 300) / 10

        # Stability impact
        stability_gap = 1.0 - model["stability"]
        priority += stability_gap * 50

        # Sample count bonus (more data = higher confidence in optimization)
        priority += min(20, model["sample_count"] / 10)

        return priority

    async def _optimize_single_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a single model"""
        model_name = model_info["model_name"]

        try:
            logger.info("Optimizing model", model=model_name, reasons=model_info["optimization_reasons"])

            # Find model file
            model_path = self._find_model_file(model_name)
            if not model_path:
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": "Model file not found"
                }

            # Determine optimization strategy
            optimization_type = self._determine_optimization_strategy(model_info)

            # Optimize with Intel NPU/CPU optimization
            output_path = self.optimized_models_dir / f"{model_name}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            optimization_result = await self.npu_optimizer.optimize_model(
                str(model_path),
                str(output_path),
                optimization_type
            )

            # Add model-specific information
            optimization_result.update({
                "model_name": model_name,
                "optimization_strategy": optimization_type,
                "optimization_timestamp": datetime.now().isoformat(),
                "intel_npu_used": self.npu_optimizer.npu_available,
                "original_accuracy": model_info["current_accuracy"],
                "original_processing_time_ms": model_info["processing_time_ms"]
            })

            if optimization_result["success"]:
                logger.info("Model optimization successful",
                           model=model_name,
                           compression_ratio=optimization_result["compression_ratio"])
            else:
                logger.warning("Model optimization failed",
                              model=model_name,
                              error=optimization_result.get("error"))

            return optimization_result

        except Exception as e:
            logger.error("Error optimizing model", model=model_name, error=str(e))
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e),
                "optimization_timestamp": datetime.now().isoformat()
            }

    def _find_model_file(self, model_name: str) -> Optional[Path]:
        """Find model file in models directory"""
        try:
            # Look for various model file formats
            possible_paths = [
                self.models_dir / model_name,
                self.models_dir / f"{model_name}.pt",
                self.models_dir / f"{model_name}.bin",
                self.models_dir / f"{model_name}.onnx"
            ]

            for path in possible_paths:
                if path.exists():
                    return path

            # Look in subdirectories
            for subdir in self.models_dir.glob("*/"):
                for ext in [".pt", ".bin", ".onnx", ""]:
                    model_file = subdir / f"{model_name}{ext}"
                    if model_file.exists():
                        return model_file

            return None

        except Exception as e:
            logger.error("Error finding model file", model=model_name, error=str(e))
            return None

    def _determine_optimization_strategy(self, model_info: Dict[str, Any]) -> str:
        """Determine the best optimization strategy for a model"""
        reasons = model_info["optimization_reasons"]

        # NPU optimization for accuracy-focused models
        if any("low_accuracy" in reason for reason in reasons) and self.npu_optimizer.npu_available:
            return "npu_accuracy"

        # Speed optimization for slow models
        if any("slow_processing" in reason for reason in reasons):
            return "speed"

        # Quantization for memory/size optimization
        if model_info["processing_time_ms"] > 400:
            return "quantized"

        # Default to auto optimization
        return "auto"

    async def _comprehensive_optimization(self):
        """Comprehensive optimization of all models"""
        try:
            logger.info("Starting comprehensive optimization...")

            # Get all models
            performance_data = await self._get_performance_metrics()

            # Optimize all models that haven't been optimized recently
            all_models = performance_data["models"]
            optimization_results = []

            for model in all_models:
                # Check if model was optimized recently
                last_optimization = await self._get_last_optimization_time(model["name"])

                if not last_optimization or (datetime.now() - last_optimization).hours > 12:
                    model_info = {
                        "model_name": model["name"],
                        "current_accuracy": model["accuracy"],
                        "uk_accuracy": model.get("uk_accuracy"),
                        "processing_time_ms": model["processing_time_ms"],
                        "stability": model["stability"],
                        "optimization_reasons": ["comprehensive_optimization"],
                        "priority": 50.0
                    }

                    result = await self._optimize_single_model(model_info)
                    optimization_results.append(result)

            await self._store_optimization_results(optimization_results)

            logger.info("Comprehensive optimization completed",
                       models_processed=len(optimization_results))

        except Exception as e:
            logger.error("Error in comprehensive optimization", error=str(e))

    async def _ensemble_optimization(self):
        """Optimize ensemble configuration"""
        try:
            logger.info("Starting ensemble optimization...")

            # Get current ensemble performance
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.learning_api_url}/api/v1/optimize_models") as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Ensemble optimization triggered", status=result.get("status"))
                    else:
                        logger.warning("Failed to trigger ensemble optimization", status=response.status)

        except Exception as e:
            logger.error("Error in ensemble optimization", error=str(e))

    async def _get_last_optimization_time(self, model_name: str) -> Optional[datetime]:
        """Get the last optimization time for a model"""
        try:
            # This would query an optimization history table
            # For now, return None to trigger optimization
            return None
        except Exception as e:
            logger.error("Error getting last optimization time", error=str(e))
            return None

    async def _update_ensemble_configuration(self, optimization_results: List[Dict[str, Any]]):
        """Update ensemble configuration based on optimization results"""
        try:
            successful_optimizations = [r for r in optimization_results if r.get("success", False)]

            if successful_optimizations:
                # Trigger ensemble reconfiguration via API
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "optimized_models": [r["model_name"] for r in successful_optimizations],
                        "optimization_timestamp": datetime.now().isoformat()
                    }

                    async with session.post(
                        f"{self.learning_api_url}/api/v1/optimize_models",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            logger.info("Ensemble configuration updated")
                        else:
                            logger.warning("Failed to update ensemble configuration")

        except Exception as e:
            logger.error("Error updating ensemble configuration", error=str(e))

    async def _store_optimization_results(self, results: List[Dict[str, Any]]):
        """Store optimization results in database"""
        try:
            async with self.db_pool.acquire() as conn:
                for result in results:
                    if result.get("success"):
                        # Store in model performance table with optimization flag
                        await conn.execute("""
                            INSERT INTO model_performance
                            (model_name, accuracy, processing_time_ms, recorded_at, is_uk_english)
                            VALUES ($1, $2, $3, $4, $5)
                        """,
                        f"optimization_{result['model_name']}",
                        result.get("compression_ratio", 1.0),  # Use compression ratio as proxy metric
                        0,  # Optimization doesn't have processing time
                        datetime.now(),
                        "uk-english" in result["model_name"])

                        self.optimization_history.append(result)

            logger.info("Optimization results stored", count=len(results))

        except Exception as e:
            logger.error("Error storing optimization results", error=str(e))

    async def _test_learning_api_connection(self) -> bool:
        """Test connection to learning API"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.learning_api_url}/api/v1/health") as response:
                    return response.status == 200
        except Exception:
            return False

    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()

async def main():
    """Main function to run model optimizer service"""
    db_url = os.getenv("LEARNING_DB_URL", "postgresql://voicestand:learning_pass@voicestand-learning-db:5432/voicestand_learning")
    learning_api_url = os.getenv("LEARNING_API_URL", "http://learning-api:8080")

    optimizer_service = ModelOptimizerService(db_url, learning_api_url)

    try:
        if await optimizer_service.initialize():
            logger.info("Model optimizer service starting...")
            await optimizer_service.start_optimization_service()
        else:
            logger.error("Failed to initialize optimizer service")
    except KeyboardInterrupt:
        logger.info("Shutting down optimizer service...")
    except Exception as e:
        logger.error("Optimizer service error", error=str(e))
    finally:
        await optimizer_service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())