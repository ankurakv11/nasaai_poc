"""
Memory Cleanup Middleware

This middleware forces garbage collection after each request to ensure
memory is freed promptly, especially important when processing large images
and running ML models with multiple concurrent users.
"""
import gc
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Try to import torch for GPU memory cleanup
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryCleanupMiddleware(BaseHTTPMiddleware):
    """
    Middleware that performs garbage collection after each request.

    This helps prevent memory bloat when processing multiple image uploads
    and running MediaPipe/U2Net models concurrently.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process the request and clean up memory afterwards.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the route handler
        """
        # Process the request
        response = await call_next(request)

        # Force garbage collection to free memory from images and model outputs
        gc.collect()

        # Clear PyTorch cache if CUDA is available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log memory cleanup (only for debug mode)
        logger.debug(f"Memory cleanup completed for {request.url.path}")

        return response
