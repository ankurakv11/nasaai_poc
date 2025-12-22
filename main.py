"""
FastAPI Application for Standalone Clothing Size Recommendation System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.routes import measurement, recommendation, body_measurement
from app.middleware.memory_cleanup import MemoryCleanupMiddleware

# Global model cache
u2net_model = None
mediapipe_pose = None  # Cached MediaPipe Pose instance for memory efficiency

# Initialize FastAPI app
app = FastAPI(
    title="Standalone Sizing API",
    description="Self-contained API for clothing size recommendation based on body measurements using MediaPipe and U2Net neural networks",
    version="2.0.0",
    contact={
        "name": "Sizing API",
        "url": "https://github.com/EhteZafar/Dress-Size-Recommender",
    },
    license_info={
        "name": "MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add memory cleanup middleware to free resources after each request
app.add_middleware(MemoryCleanupMiddleware)

# Mount static files for uploads
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(measurement.router, prefix="", tags=["measurements"])
app.include_router(recommendation.router, prefix="", tags=["recommendations"])
app.include_router(body_measurement.router, prefix="", tags=["body-measurements"])

@app.on_event("startup")
async def load_models():
    """Load U2Net and MediaPipe models once at startup to avoid loading per request"""
    global u2net_model, mediapipe_pose
    try:
        print("Loading U2Net model at startup...")
        from app.utils.libs.networks import U2NET
        u2net_model = U2NET("u2net")
        print("U2Net model loaded successfully and cached for reuse")
    except Exception as e:
        print(f"Warning: Failed to load U2Net model at startup: {e}")
        print("Background removal will use fallback methods")
        u2net_model = None

    try:
        print("Loading MediaPipe Pose model at startup...")
        import mediapipe as mp
        mediapipe_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,  # Lightweight model for memory efficiency
            enable_segmentation=False,  # Disable segmentation to save memory
            min_detection_confidence=0.5
        )
        print("MediaPipe Pose model loaded successfully and cached for reuse")
    except Exception as e:
        print(f"Warning: Failed to load MediaPipe model at startup: {e}")
        print("Body measurements will create pose instances per request")
        mediapipe_pose = None

def get_u2net_model():
    """Get the cached U2Net model instance"""
    return u2net_model

def get_mediapipe_pose():
    """Get the cached MediaPipe Pose instance"""
    return mediapipe_pose

@app.get("/")
async def root():
    """
    Root endpoint with API information
    
    Returns comprehensive information about the standalone Sizing API including
    available endpoints, features, and system status.
    """
    return {
        "message": "Standalone Sizing API is running",
        "version": "2.0.0",
        "description": "Self-contained clothing size recommendation system",
        "features": [
            "MediaPipe body measurement extraction",
            "U2Net neural network background removal", 
            "Anthropometric size analysis",
            "Multi-format image processing",
            "Real-time size recommendations"
        ],
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "process_images": "/process-images",
            "get_measurements": "/measurements", 
            "save_measurements": "/measurements",
            "get_categories": "/categories",
            "get_products": "/categories/{category}/products",
            "get_recommendations": "/recommendations",
            "try_virtually": "/try-virtually",
            "analyze_fit": "/analyze-fit",
            "best_fit": "/best-fit/{product}",
            "body_measurements": "/body-measurements",
            "body_measurements_from_urls": "/body-measurements-from-urls"
        },
        "status": "standalone - no external dependencies"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns system health status including model availability and processing capabilities.
    """
    # Check if models are available
    model_path = os.path.join("app", "models", "u2net", "u2net.pth")
    model_file_exists = os.path.exists(model_path)
    model_loaded = u2net_model is not None
    
    # Check GPU availability if torch is available
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Check memory if psutil is available
    memory_info = "N/A"
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_info = f"{memory.percent}% used"
    except ImportError:
        pass
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "system": {
            "standalone": True,
            "memory_usage": memory_info,
            "gpu_available": gpu_available,
            "model_file_exists": model_file_exists,
            "model_loaded_in_memory": model_loaded
        },
        "capabilities": {
            "background_removal": model_loaded,
            "body_measurement": True,
            "size_analysis": True,
            "multi_format_images": True
        }
    }

@app.get("/api-info")
async def api_info():
    """
    Comprehensive API information and capabilities
    
    Provides detailed information about the standalone sizing API including
    technical specifications, processing capabilities, and usage guidelines.
    """
    return {
        "api": {
            "name": "Standalone Sizing API",
            "version": "2.0.0",
            "description": "Self-contained body measurement and size recommendation system",
            "type": "standalone",
            "dependencies": "none"
        },
        "features": {
            "image_processing": {
                "background_removal": "U2Net neural network",
                "supported_formats": ["JPEG", "PNG", "JPG", "BMP"],
                "max_resolution": "Automatically optimized",
                "processing_time": "2-5 seconds per image pair"
            },
            "body_measurement": {
                "technology": "MediaPipe pose detection",
                "measurements": ["height", "chest", "waist", "hip", "shoulder", "neck", "inseam"],
                "accuracy": "±2cm for most measurements",
                "gender_support": ["male", "female"],
                "body_types": ["Triangle", "Diamond", "Inverted", "Rectangle", "Hourglass"]
            },
            "size_analysis": {
                "algorithm": "Anthropometric analysis",
                "scoring": "0-10 scale with detailed feedback",
                "fit_types": ["Best Fit", "Regular Fit", "Tight Fit", "Loose Fit", "Poor Fit"],
                "categories": "Configurable via size charts"
            }
        },
        "technical_specs": {
            "python_version": "3.8+",
            "framework": "FastAPI",
            "ai_models": ["U2Net", "MediaPipe"],
            "storage": "Local file system",
            "gpu_support": "Optional (CUDA)",
            "memory_requirements": "4GB+ recommended"
        },
        "deployment": {
            "type": "standalone",
            "portability": "fully portable",
            "external_dependencies": "none",
            "database": "not required",
            "internet": "not required for processing"
        },
        "privacy": {
            "data_processing": "local only",
            "image_storage": "temporary",
            "external_apis": "none",
            "data_retention": "user controlled"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)