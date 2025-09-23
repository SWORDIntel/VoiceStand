"""
VoiceStand Learning API
FastAPI service for the advanced learning system providing real-time model optimization
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from learning_engine import LearningEngine, RecognitionResult, LearningPattern
from pattern_analyzer import PatternAnalyzer
from uk_specializer import UKSpecializer
from model_optimizer import ModelOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
learning_engine: Optional[LearningEngine] = None
pattern_analyzer: Optional[PatternAnalyzer] = None
uk_specializer: Optional[UKSpecializer] = None
model_optimizer: Optional[ModelOptimizer] = None

# Pydantic models for API
class AudioSample(BaseModel):
    audio_data: List[float] = Field(..., description="Raw audio samples")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    duration_ms: int = Field(..., description="Duration in milliseconds")
    speaker_id: Optional[str] = Field(None, description="Speaker identifier")

class RecognitionRequest(BaseModel):
    audio: AudioSample
    ground_truth: Optional[str] = Field(None, description="Known correct transcription for training")
    context: Optional[str] = Field(None, description="Context information")
    is_uk_english: bool = Field(default=True, description="UK English specialization")

class RecognitionResponse(BaseModel):
    text: str
    confidence: float
    model_used: str
    processing_time_ms: int
    speaker_id: Optional[str] = None
    ensemble_agreement: float
    improvements_applied: List[str]

class TrainingData(BaseModel):
    audio_samples: List[AudioSample]
    transcriptions: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OptimizationRequest(BaseModel):
    target_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    focus_area: str = Field(default="uk_english", description="Optimization focus")
    max_training_time_minutes: int = Field(default=60, ge=1, le=480)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    performance_metrics: Dict[str, float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global learning_engine, pattern_analyzer, uk_specializer, model_optimizer

    logger.info("Starting VoiceStand Learning API...")

    # Initialize database connection
    db_url = os.getenv("LEARNING_DB_URL", "postgresql://voicestand:learning_pass@localhost:5433/voicestand_learning")

    try:
        # Initialize core components
        learning_engine = LearningEngine(db_url)
        await learning_engine.initialize()

        pattern_analyzer = PatternAnalyzer(learning_engine)
        uk_specializer = UKSpecializer(learning_engine)
        model_optimizer = ModelOptimizer(learning_engine)

        # Warm up services
        await pattern_analyzer.initialize()
        await uk_specializer.initialize()
        await model_optimizer.initialize()

        logger.info("Learning API services initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize learning services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down Learning API...")
        if learning_engine:
            await learning_engine.close()

# Create FastAPI app
app = FastAPI(
    title="VoiceStand Learning API",
    description="Advanced machine learning API for continuous voice recognition improvement",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    services_status = {
        "learning_engine": "healthy" if learning_engine else "unhealthy",
        "pattern_analyzer": "healthy" if pattern_analyzer else "unhealthy",
        "uk_specializer": "healthy" if uk_specializer else "unhealthy",
        "model_optimizer": "healthy" if model_optimizer else "unhealthy",
    }

    # Get performance metrics
    performance_metrics = {}
    if learning_engine:
        try:
            performance_metrics = await learning_engine.get_performance_metrics()
        except Exception as e:
            logger.warning(f"Could not fetch performance metrics: {e}")
            performance_metrics = {"error": "metrics_unavailable"}

    overall_status = "healthy" if all(status == "healthy" for status in services_status.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        services=services_status,
        performance_metrics=performance_metrics
    )

# Core recognition endpoint
@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_audio(request: RecognitionRequest, background_tasks: BackgroundTasks):
    """Process audio and return recognition results with learning"""
    if not learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not initialized")

    start_time = time.time()

    try:
        # Process recognition with ensemble models
        result = await learning_engine.process_audio(
            audio_data=request.audio.audio_data,
            sample_rate=request.audio.sample_rate,
            ground_truth=request.ground_truth,
            is_uk_english=request.is_uk_english,
            speaker_id=request.audio.speaker_id
        )

        # Apply UK specializations if needed
        if request.is_uk_english and uk_specializer:
            result = await uk_specializer.enhance_recognition(result)

        # Analyze patterns for learning
        if pattern_analyzer and request.ground_truth:
            background_tasks.add_task(
                pattern_analyzer.analyze_recognition_patterns,
                result,
                request.ground_truth
            )

        processing_time = int((time.time() - start_time) * 1000)

        return RecognitionResponse(
            text=result.text,
            confidence=result.confidence,
            model_used=result.model_used,
            processing_time_ms=processing_time,
            speaker_id=result.speaker_id,
            ensemble_agreement=result.ensemble_agreement,
            improvements_applied=getattr(result, 'improvements_applied', [])
        )

    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recognition processing failed: {str(e)}")

# Training endpoint
@app.post("/train")
async def train_models(data: TrainingData, background_tasks: BackgroundTasks):
    """Submit training data for continuous learning"""
    if not learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not initialized")

    try:
        # Queue training data
        training_job_id = await learning_engine.queue_training_data(
            audio_samples=[sample.dict() for sample in data.audio_samples],
            transcriptions=data.transcriptions,
            metadata=data.metadata
        )

        # Start background training
        background_tasks.add_task(learning_engine.process_training_queue)

        return {
            "message": "Training data queued successfully",
            "job_id": training_job_id,
            "estimated_completion": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Training submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Model optimization endpoint
@app.post("/optimize")
async def optimize_models(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Trigger model optimization process"""
    if not model_optimizer:
        raise HTTPException(status_code=503, detail="Model optimizer not initialized")

    try:
        optimization_id = await model_optimizer.start_optimization(
            target_accuracy=request.target_accuracy,
            focus_area=request.focus_area,
            max_time_minutes=request.max_training_time_minutes
        )

        background_tasks.add_task(model_optimizer.run_optimization, optimization_id)

        return {
            "message": "Model optimization started",
            "optimization_id": optimization_id,
            "target_accuracy": request.target_accuracy,
            "estimated_completion_time": f"{request.max_training_time_minutes} minutes"
        }

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

# Analytics endpoints
@app.get("/analytics/patterns")
async def get_learning_patterns(limit: int = 100, pattern_type: Optional[str] = None):
    """Get discovered learning patterns"""
    if not pattern_analyzer:
        raise HTTPException(status_code=503, detail="Pattern analyzer not initialized")

    try:
        patterns = await pattern_analyzer.get_patterns(limit=limit, pattern_type=pattern_type)
        return {"patterns": [pattern.__dict__ for pattern in patterns]}
    except Exception as e:
        logger.error(f"Failed to fetch patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch patterns: {str(e)}")

@app.get("/analytics/performance")
async def get_performance_analytics():
    """Get system performance analytics"""
    if not learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not initialized")

    try:
        metrics = await learning_engine.get_detailed_analytics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to fetch analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics unavailable: {str(e)}")

@app.get("/models/status")
async def get_model_status():
    """Get current model ensemble status"""
    if not learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not initialized")

    try:
        status = await learning_engine.get_model_status()
        return status
    except Exception as e:
        logger.error(f"Failed to fetch model status: {e}")
        raise HTTPException(status_code=500, detail=f"Model status unavailable: {str(e)}")

# Error handlers
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )