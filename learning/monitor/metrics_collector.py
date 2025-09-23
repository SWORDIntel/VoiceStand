"""
VoiceStand Learning System - Metrics Collector
Collects and processes learning system metrics
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and processes learning metrics"""

    def __init__(self):
        self.metrics_buffer = []
        self.collection_active = False

    async def start_collection(self):
        """Start metrics collection"""
        self.collection_active = True
        logger.info("Starting metrics collection...")

        while self.collection_active:
            try:
                await self.process_metrics_buffer()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)

    async def process_metrics_buffer(self):
        """Process collected metrics"""
        if not self.metrics_buffer:
            return

        # Process metrics in batches
        batch_size = 100
        for i in range(0, len(self.metrics_buffer), batch_size):
            batch = self.metrics_buffer[i:i + batch_size]
            await self.process_batch(batch)

        # Clear processed metrics
        self.metrics_buffer.clear()

    async def process_batch(self, batch: List[Dict]):
        """Process a batch of metrics"""
        # Calculate aggregated metrics
        total_accuracy = sum(m.get('accuracy', 0) for m in batch)
        avg_accuracy = total_accuracy / len(batch) if batch else 0

        logger.info(f"Processed {len(batch)} metrics, avg accuracy: {avg_accuracy:.2%}")

    def add_metric(self, metric: Dict):
        """Add a metric to the collection buffer"""
        self.metrics_buffer.append({
            **metric,
            'collected_at': datetime.utcnow().isoformat()
        })