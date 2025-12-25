"""
Pydantic schemas for request and response models
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional, Union


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    status_code: int = 400


class ProcessImageRequest(BaseModel):
    """Request model for image processing"""
    gender: str = Field(..., description="Gender of the person in the image (male/female)")
    height_cm: float = Field(..., description="Height of the person in centimeters")
    weight_kg: Optional[float] = Field(None, description="Weight of the person in kilograms")


class ProcessImageResponse(BaseModel):
    """Response model for image processing"""
    measurements: Dict[str, float]
    processed_front_image: Optional[str] = None
    processed_side_image: Optional[str] = None
    body_type: Optional[str] = None
    message: str = "Images processed successfully"


class ProcessImagesFromUrlsRequest(BaseModel):
    """Request model for processing images from URLs"""
    front_image_url: HttpUrl
    side_image_url: HttpUrl
    gender: str = Field(..., description="Gender of the person in the image (male/female)")
    height_cm: float = Field(..., description="Height of the person in centimeters")
    height: Optional[float] = None  # Added for frontend compatibility
    weight_kg: Optional[float] = Field(None, description="Weight of the person in kilograms")
    weight: Optional[float] = None  # Added for frontend compatibility
    preferred_size: Optional[str] = None
    body_type: Optional[str] = None


class UserMeasurements(BaseModel):
    """User measurements model - all measurements in INCHES except height in CM"""
    height: float  # Height in CM
    neck: Optional[float] = None  # Neck circumference in inches
    chest: Optional[float] = None  # Chest/Bust circumference in inches
    waist: Optional[float] = None  # Waist circumference in inches
    hip: Optional[float] = None  # Hip circumference in inches (accepts both 'hip' and 'Butt')
    Butt: Optional[float] = None  # Hip circumference in inches (legacy field name)
    shoulder_width: Optional[float] = None  # Shoulder width in inches
    shoulder: Optional[float] = None  # Shoulder width in inches (alias for shoulder_width)
    body_length: Optional[float] = None  # Body/Torso length in inches
    inseam: Optional[float] = None  # Inseam length in inches
    gender: Optional[str] = None  # Gender (M/F)
    weight: Optional[float] = None  # Weight in KG
    body_type: Optional[str] = None  # Body type for females
    
    class Config:
        # Allow None values for optional fields
        extra = "allow"


class VirtualTryOnRequest(BaseModel):
    """Request model for virtual try-on"""
    user_measurements: UserMeasurements
    product_id: str
    size: str


class SizeChartItem(BaseModel):
    """Size chart item model"""
    size_label: str
    chest_inches: Optional[float] = None
    waist_inches: Optional[float] = None
    shoulder_inches: Optional[float] = None
    hips_inches: Optional[float] = None
    length_inches: Optional[float] = None
    inseam_inches: Optional[float] = None


class RecommendationRequest(BaseModel):
    """Request model for size recommendation"""
    user_measurements: UserMeasurements
    product_category: str
    brand: Optional[str] = None
    fit_preference: Optional[str] = "regular"
    size_chart: Optional[List[SizeChartItem]] = None


class SizeRecommendation(BaseModel):
    """Size recommendation model"""
    size: str
    rating: Optional[str] = None
    score: Optional[float] = None
    product: Optional[str] = None
    fit_category: Optional[str] = None
    measurements_match: Optional[Dict[str, float]] = None


class RecommendationResponse(BaseModel):
    """Response model for size recommendation"""
    categories: List[str]
    products: Dict[str, List[str]]
    recommendations: List[SizeRecommendation]
    recommended_size: Optional[str] = None
    fit_score: Optional[float] = None
    alternative_sizes: Optional[Dict[str, float]] = None
    measurements: Optional[Dict[str, Any]] = None
    message: Optional[str] = "Size recommendations based on your measurements"
    
    
class GarmentRequest(BaseModel):
    """Request model for garment information"""
    category: str
    brand: Optional[str] = None
    product_id: Optional[str] = None
    size: Optional[str] = None
    
    
class FitAnalysisResponse(BaseModel):
    """Response model for fit analysis"""
    fit_score: float
    fit_category: str
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    message: str
    
    
class FitAnalysisDetail(BaseModel):
    """Detailed fit analysis model"""
    measurement_name: str
    user_value: float
    garment_value: float
    difference: float
    fit_score: float
    comment: str


class MeasurementsResponse(BaseModel):
    """Response model for user measurements only"""
    measurements: Dict[str, float]
    message: str = "Body measurements extracted successfully"


class SizeRecommendationsResponse(BaseModel):
    """Response model for size recommendations only"""
    recommended_size: Optional[str] = None
    fit_score: Optional[float] = None
    alternative_sizes: Optional[Dict[str, float]] = {}
    message: str = "Size recommendations based on your measurements"


class SizeRecommendationsRequest(BaseModel):
    """Request model for size recommendations"""
    measurements: Dict[str, float]
    product_category: str
    product_id: Optional[str] = None
    brand: Optional[str] = None
    fit_preference: Optional[str] = "regular"


class BodyMeasurementRequest(BaseModel):
    """Request model for body measurements from URLs"""
    front_image_url: HttpUrl
    side_image_url: HttpUrl
    gender: str = Field(..., description="Gender of the person in the image (male/female)")
    height: float = Field(..., description="Height of the person in centimeters")
    weight: Optional[float] = Field(None, description="Weight of the person in kilograms")
    body_type: Optional[str] = None


class BodyMeasurementResponse(BaseModel):
    """Response model for body measurements only"""
    success: bool = True
    message: str
    measurements: UserMeasurements
    processed_images: Dict[str, str]
    annotated_images: Dict[str, str]