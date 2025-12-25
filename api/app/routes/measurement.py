"""
API routes for measurement operations
Standalone version - no external dependencies
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import os
import traceback

from ..models.schemas import (
    ProcessImageRequest, ProcessImageResponse, UserMeasurements, ErrorResponse,
    ProcessImagesFromUrlsRequest, VirtualTryOnRequest
)
from ..services.image_processing import image_service
from ..services.measurement import measurement_service
from ..services.url_image_processor import url_image_processor

router = APIRouter()

@router.post("/process-images", response_model=ProcessImageResponse)
async def process_images(
    front_image: UploadFile = File(..., description="Front view image"),
    side_image: UploadFile = File(..., description="Side view image"),
    height: float = Form(..., description="Height in cm"),
    weight: float = Form(..., description="Weight in kg"),
    gender: str = Form(..., description="Gender (M/F)"),
    preferred_size: Optional[str] = Form(None, description="Preferred size"),
    body_type: Optional[str] = Form(None, description="Body type for females (Triangle, Diamond, Inverted, Rectangle, Hourglass)")
):
    """
    Process uploaded images and extract body measurements using MediaPipe and U2Net
    
    This endpoint performs comprehensive image processing including:
    - U2Net neural network background removal for cleaner analysis
    - MediaPipe pose detection for accurate body landmark identification
    - Anthropometric measurement calculation based on pose landmarks
    - Support for multiple body types and gender-specific analysis
    
    The system is completely self-contained with no external API dependencies.
    """
    try:
        # Validate inputs
        if gender.upper() not in ['M', 'F']:
            raise HTTPException(status_code=400, detail="Gender must be M or F")
        
        # Validate body type for females
        if gender.upper() == 'F' and body_type:
            valid_body_types = ['Triangle', 'Diamond', 'Inverted', 'Rectangle', 'Hourglass']
            if body_type not in valid_body_types:
                raise HTTPException(status_code=400, detail=f"Body type must be one of: {', '.join(valid_body_types)}")
        
        if height <= 0 or height > 300:
            raise HTTPException(status_code=400, detail="Height must be between 0 and 300 cm")
        
        if weight <= 0 or weight > 500:
            raise HTTPException(status_code=400, detail="Weight must be between 0 and 500 kg")
        
        # Check file types
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        front_ext = os.path.splitext(front_image.filename)[1].lower()
        side_ext = os.path.splitext(side_image.filename)[1].lower()
        
        if front_ext not in allowed_extensions or side_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Only JPG, PNG, and BMP files are allowed")
        
        # Save uploaded files
        front_content = await front_image.read()
        side_content = await side_image.read()
        
        front_path = image_service.save_uploaded_file(front_content, front_image.filename)
        side_path = image_service.save_uploaded_file(side_content, side_image.filename)
        
        # Process images
        processed_front, processed_side = image_service.process_image_pair(front_path, side_path)
        
        if not processed_front or not processed_side:
            error_msg = f"Image processing failed - Front: {'OK' if processed_front else 'FAILED'}, Side: {'OK' if processed_side else 'FAILED'}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Get measurements
        measurements = measurement_service.get_measurements_from_images(
            processed_front, processed_side, height, gender.upper(), weight, preferred_size, body_type
        )
        
        # Convert to response model
        user_measurements = UserMeasurements(**measurements)
        
        return ProcessImageResponse(
            success=True,
            message="Images processed and measurements extracted successfully",
            measurements=user_measurements,
            processed_images={
                "front": processed_front,
                "side": processed_side
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in process_images: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/measurements", response_model=UserMeasurements)
async def get_stored_measurements():
    """
    Get the most recently stored measurements
    
    Retrieves measurements from local storage. All data is processed and stored
    locally with no external dependencies or cloud storage requirements.
    """
    try:
        measurements = measurement_service.load_measurements()
        
        if not measurements:
            raise HTTPException(status_code=404, detail="No measurements found. Please process images first.")
        
        if not measurement_service.validate_measurements(measurements):
            raise HTTPException(status_code=422, detail="Invalid measurements data found")
        
        return UserMeasurements(**measurements)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_stored_measurements: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve measurements: {str(e)}")

@router.post("/measurements", response_model=UserMeasurements)
async def save_measurements(measurements: UserMeasurements):
    """
    Save custom measurements manually
    """
    try:
        measurements_dict = measurements.dict()
        
        if not measurement_service.validate_measurements(measurements_dict):
            raise HTTPException(status_code=422, detail="Invalid measurements provided")
        
        measurement_service.save_measurements(measurements_dict)
        
        return measurements
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in save_measurements: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save measurements: {str(e)}")

@router.delete("/measurements")
async def clear_measurements():
    """
    Clear stored measurements
    """
    try:
        if os.path.exists(measurement_service.data_file):
            os.remove(measurement_service.data_file)
        
        return {"message": "Measurements cleared successfully"}
        
    except Exception as e:
        print(f"Error in clear_measurements: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear measurements: {str(e)}")

@router.post("/process-images-from-urls", response_model=ProcessImageResponse)
async def process_images_from_urls(request: ProcessImagesFromUrlsRequest):
    """
    Process images from URLs using U2Net and MediaPipe
    
    Downloads images from the provided URLs and processes them through the complete
    U2Net background removal and MediaPipe measurement extraction pipeline.
    """
    try:
        # Validate inputs
        if request.gender.upper() not in ['M', 'F']:
            raise HTTPException(status_code=400, detail="Gender must be M or F")
        
        # Validate body type for females
        if request.gender.upper() == 'F' and request.body_type:
            valid_body_types = ['Triangle', 'Diamond', 'Inverted', 'Rectangle', 'Hourglass']
            if request.body_type not in valid_body_types:
                raise HTTPException(status_code=400, detail=f"Body type must be one of: {', '.join(valid_body_types)}")
        
        if request.height <= 0 or request.height > 300:
            raise HTTPException(status_code=400, detail="Height must be between 0 and 300 cm")
        
        if request.weight and (request.weight <= 0 or request.weight > 500):
            raise HTTPException(status_code=400, detail="Weight must be between 0 and 500 kg")
        
        # Process images from URLs
        measurements = url_image_processor.process_images_from_urls(
            request.front_image_url,
            request.side_image_url,
            request.height,
            request.gender.upper(),
            request.weight,
            request.preferred_size,
            request.body_type
        )
        
        if not measurements:
            raise HTTPException(status_code=500, detail="Failed to process images from URLs")
        
        # Convert to response model
        user_measurements = UserMeasurements(**measurements)
        
        return ProcessImageResponse(
            success=True,
            message="Images processed from URLs and measurements extracted successfully",
            measurements=user_measurements,
            processed_images={
                "front": "processed_from_url",
                "side": "processed_from_url"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in process_images_from_urls: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/cleanup")
async def cleanup_files():
    """
    Clean up temporary and processed files
    
    Removes temporary files created during image processing to free up disk space.
    This helps maintain optimal performance in standalone deployments.
    """
    try:
        image_service.cleanup_temp_files()
        return {"message": "Cleanup completed successfully"}
        
    except Exception as e:
        print(f"Error in cleanup_files: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")