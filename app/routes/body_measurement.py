"""
API routes for body measurement operations only
No garment fitting or size recommendations
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import os
import traceback

from ..models.schemas import (
    ProcessImageRequest, BodyMeasurementResponse, UserMeasurements, ErrorResponse,
    ProcessImagesFromUrlsRequest, BodyMeasurementRequest
)
from ..services.image_processing import image_service
from ..services.measurement import measurement_service
from ..services.url_image_processor import url_image_processor

router = APIRouter()

@router.post("/body-measurements", response_model=BodyMeasurementResponse)
async def get_body_measurements(
    front_image: UploadFile = File(..., description="Front view image"),
    side_image: UploadFile = File(..., description="Side view image"),
    height: float = Form(..., description="Height in cm"),
    weight: float = Form(..., description="Weight in kg"),
    gender: str = Form(..., description="Gender (M/F)"),
    body_type: Optional[str] = Form(None, description="Body type for females (Triangle, Diamond, Inverted, Rectangle, Hourglass)")
):
    """
    Process uploaded images and extract body measurements using MediaPipe and U2Net
    
    This endpoint performs comprehensive image processing including:
    - U2Net neural network background removal for cleaner analysis
    - MediaPipe pose detection for accurate body landmark identification
    - Anthropometric measurement calculation based on pose landmarks
    - Support for multiple body types and gender-specific analysis
    
    Returns only body measurements without garment fitting analysis
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

        # Free uploaded bytes after saving to prevent memory bloat
        del front_content, side_content
        import gc
        gc.collect()

        # Process images
        processed_front, processed_side = image_service.process_image_pair(front_path, side_path)
        
        if not processed_front or not processed_side:
            error_msg = f"Image processing failed - Front: {'OK' if processed_front else 'FAILED'}, Side: {'OK' if processed_side else 'FAILED'}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Get measurements without preferred size (not needed for body measurements only)
        measurements = measurement_service.get_measurements_from_images(
            processed_front, processed_side, height, gender.upper(), weight, None, body_type
        )
        
        # Add gender and weight to measurements for response
        measurements['gender'] = gender.upper()
        if weight:
            measurements['weight'] = weight
        if body_type:
            measurements['body_type'] = body_type
            
        # Ensure 'hip' field exists (mediapipe returns 'Butt')
        if 'Butt' in measurements and 'hip' not in measurements:
            measurements['hip'] = measurements['Butt']
        
        # Ensure 'shoulder' field exists (mediapipe returns 'shoulder_width')
        if 'shoulder_width' in measurements and 'shoulder' not in measurements:
            measurements['shoulder'] = measurements['shoulder_width']
        
        # Convert to response model (all measurements are in INCHES)
        user_measurements = UserMeasurements(**measurements)
        
        return BodyMeasurementResponse(
            success=True,
            message="Images processed and body measurements extracted successfully",
            measurements=user_measurements,
            processed_images={
                "front": processed_front,
                "side": processed_side
            },
            annotated_images={
                "front": processed_front.replace("_processed.png", "_annotated.jpg"),
                "side": processed_side.replace("_processed.png", "_annotated.jpg")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_body_measurements: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/body-measurements-from-urls", response_model=BodyMeasurementResponse)
async def get_body_measurements_from_urls(request: BodyMeasurementRequest):
    """
    Process images from URLs and extract body measurements
    
    Downloads images from the provided URLs and processes them through the complete
    U2Net background removal and MediaPipe measurement extraction pipeline.
    
    Returns only body measurements without garment fitting analysis
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
            None,  # No preferred size needed
            request.body_type
        )
        
        if not measurements:
            raise HTTPException(status_code=500, detail="Failed to process images from URLs")
        
        # Add gender and weight to measurements for response
        measurements['gender'] = request.gender.upper()
        if request.weight:
            measurements['weight'] = request.weight
        if request.body_type:
            measurements['body_type'] = request.body_type
            
        # Ensure 'hip' field exists (mediapipe returns 'Butt')
        if 'Butt' in measurements and 'hip' not in measurements:
            measurements['hip'] = measurements['Butt']
        
        # Ensure 'shoulder' field exists (mediapipe returns 'shoulder_width')
        if 'shoulder_width' in measurements and 'shoulder' not in measurements:
            measurements['shoulder'] = measurements['shoulder_width']
        
        # Convert to response model (all measurements are in INCHES)
        user_measurements = UserMeasurements(**measurements)
        
        # Get paths to processed and annotated images
        processed_front = os.path.join("uploads", "processed", "front_processed.png")
        processed_side = os.path.join("uploads", "processed", "side_processed.png")
        annotated_front = os.path.join("uploads", "processed", "front_annotated.jpg")
        annotated_side = os.path.join("uploads", "processed", "side_annotated.jpg")
        
        return BodyMeasurementResponse(
            success=True,
            message="Images processed from URLs and body measurements extracted successfully",
            measurements=user_measurements,
            processed_images={
                "front": processed_front,
                "side": processed_side
            },
            annotated_images={
                "front": annotated_front,
                "side": annotated_side
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_body_measurements_from_urls: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
