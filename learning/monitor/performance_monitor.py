"""
VoiceStand Learning System - Performance Monitor
Real-time monitoring of learning system metrics
"""

import asyncio
import logging
import time
from typing import Dict, List
import json
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors learning system performance in real-time"""

    def __init__(self, learning_api_url: str):
        self.learning_api_url = learning_api_url
        self.metrics_history = []
        self.monitoring_active = False

    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        self.monitoring_active = True
        logger.info("Starting performance monitoring...")

        while self.monitoring_active:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)

                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                await asyncio.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)

    async def collect_metrics(self) -> Dict:
        """Collect current performance metrics"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get learning API health
                async with session.get(f"{self.learning_api_url}/api/v1/health") as resp:
                    health_data = await resp.json()

                # Get model performance
                async with session.get(f"{self.learning_api_url}/api/v1/model_performance") as resp:
                    performance_data = await resp.json()

                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'health': health_data,
                    'performance': performance_data,
                    'system_load': await self.get_system_metrics()
                }

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }

    async def get_system_metrics(self) -> Dict:
        """Get system-level metrics"""
        return {
            'cpu_usage': 0.0,  # Placeholder - integrate with psutil
            'memory_usage': 0.0,
            'disk_usage': 0.0
        }