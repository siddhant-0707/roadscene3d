"""OpenTelemetry integration for distributed tracing and structured logging."""

import logging
import json
import time
from typing import Dict, Optional
from functools import wraps

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry not available, telemetry will be disabled")

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Structured JSON logger for telemetry."""
    
    def __init__(self, name: str = __name__):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.tracer = None
        
        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        try:
            resource = Resource.create({"service.name": "roadscene3d"})
            provider = TracerProvider(resource=resource)
            
            # Export to OTLP (can be configured to export to collector)
            # For now, just set up provider without exporter (no-op)
            trace.set_tracer_provider(provider)
            
            self.tracer = trace.get_tracer(__name__)
            logger.info("OpenTelemetry tracing initialized")
        except Exception as e:
            logger.warning(f"Failed to setup OpenTelemetry: {e}")
    
    def log_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        """
        Log a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        log_data = {
            'type': 'metric',
            'name': name,
            'value': value,
            'timestamp': time.time(),
            'tags': tags or {}
        }
        self.logger.info(json.dumps(log_data))
    
    def log_event(self, event: str, attributes: Optional[Dict] = None):
        """
        Log an event.
        
        Args:
            event: Event name
            attributes: Optional event attributes
        """
        log_data = {
            'type': 'event',
            'event': event,
            'timestamp': time.time(),
            'attributes': attributes or {}
        }
        self.logger.info(json.dumps(log_data))
    
    def trace_function(self, func_name: Optional[str] = None):
        """
        Decorator to trace function execution.
        
        Args:
            func_name: Optional function name (defaults to function __name__)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                trace_name = func_name or func.__name__
                
                if self.tracer:
                    with self.tracer.start_as_current_span(trace_name):
                        start = time.time()
                        result = func(*args, **kwargs)
                        duration = time.time() - start
                        self.log_metric(f"{trace_name}.duration", duration)
                        return result
                else:
                    start = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start
                    self.log_metric(f"{trace_name}.duration", duration)
                    return result
            
            return wrapper
        return decorator


# Global structured logger instance
telemetry_logger = StructuredLogger("roadscene3d.telemetry")
