"""
API routes for size recommendation operations
Standalone version with built-in size analysis algorithms
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

from ..models.schemas import (
    UserMeasurements, GarmentRequest, RecommendationRequest, 
    RecommendationResponse, FitAnalysisResponse, SizeRecommendation,
    FitAnalysisDetail, ErrorResponse, VirtualTryOnRequest, ProcessImagesFromUrlsRequest,
    MeasurementsResponse, SizeRecommendationsResponse, SizeRecommendationsRequest
)
from ..services.recommendation import recommendation_service
from ..services.measurement import measurement_service
from ..services.url_image_processor import url_image_processor

router = APIRouter()

@router.get("/categories", response_model=List[str])
async def get_categories():
    """
    Get all available garment categories
    
    Returns categories from the built-in size chart database.
    No external dependencies required.
    """
    try:
        categories = recommendation_service.get_categories()
        return categories
        
    except Exception as e:
        print(f"Error in get_categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve categories: {str(e)}")

@router.get("/categories/{category}/products", response_model=List[str])
async def get_products_by_category(category: str):
    """
    Get all products for a specific category
    """
    try:
        products = recommendation_service.get_products_by_category(category)
        
        if not products:
            raise HTTPException(status_code=404, detail=f"No products found for category: {category}")
        
        return products
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_products_by_category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve products: {str(e)}")

@router.get("/products/{product}/sizes", response_model=List[str])
async def get_sizes_for_product(product: str):
    """
    Get all available sizes for a specific product
    """
    try:
        sizes = recommendation_service.get_sizes_for_product(product)
        
        if not sizes:
            raise HTTPException(status_code=404, detail=f"No sizes found for product: {product}")
        
        return sizes
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_sizes_for_product: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sizes: {str(e)}")

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get size recommendations based on user measurements
    
    Uses built-in anthropometric analysis algorithms to provide accurate
    size recommendations. Supports multiple body types and gender-specific
    sizing without requiring external APIs or databases.
    """
    try:
        measurements_dict = request.user_measurements.dict()
        
        # Convert measurements to cm for analysis
        measurements_cm = measurements_dict  # Already in cm, no need to convert
        
        if request.product_category:
            # Check if custom size chart is provided
            if request.size_chart and len(request.size_chart) > 0:
                # Convert Pydantic models to dictionaries
                size_chart_data = [size_data.dict() for size_data in request.size_chart]
                
                # Use custom size chart for recommendations
                recommendations_data = recommendation_service.get_recommendations_with_custom_size_chart(
                    measurements_cm, size_chart_data, request.product_category
                )
                categories = [request.product_category]
                products = {request.product_category: ["CUSTOM_001"]}
            else:
                # Get recommendations for specific category using default size chart
                recommendations_data = recommendation_service.get_recommendations_for_category(
                    measurements_cm, request.product_category
                )
                categories = [request.product_category]
                products = {request.product_category: recommendation_service.get_products_by_category(request.product_category)}
        else:
            # Get all recommendations
            all_data = recommendation_service.get_all_recommendations(measurements_cm)
            recommendations_data = all_data["recommendations"]
            categories = all_data["categories"]
            products = all_data["products"]
        
        # Convert to response format
        recommendations = [
            SizeRecommendation(
                size=rec["size"],
                rating=rec["rating"],
                score=rec["score"],
                product=rec["product"]
            )
            for rec in recommendations_data
        ]
        
        # Get the best recommended size
        best_recommendation = recommendations[0] if recommendations else None
        recommended_size = best_recommendation.size if best_recommendation else None
        fit_score = best_recommendation.score if best_recommendation else None
        
        # Create alternative sizes dictionary
        alternative_sizes = {}
        for rec in recommendations[1:]:
            alternative_sizes[rec.size] = rec.score
        
        return RecommendationResponse(
            categories=categories,
            products=products,
            recommendations=recommendations,
            recommended_size=recommended_size,
            fit_score=fit_score,
            alternative_sizes=alternative_sizes,
            measurements=measurements_dict
        )
        
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.post("/analyze-fit", response_model=FitAnalysisResponse)
async def analyze_fit(request: GarmentRequest):
    """
    Analyze fit for a specific garment
    
    Performs comprehensive fit analysis using built-in size logic algorithms.
    Provides detailed scoring for each body measurement and overall fit rating.
    All calculations are performed locally with no external dependencies.
    """
    try:
        measurements_dict = request.measurements.dict()
        
        # Convert measurements to cm for analysis
        measurements_cm = measurement_service.convert_measurements_to_cm(measurements_dict)
        
        # Get fit analysis
        analysis_result = recommendation_service.analyze_fit(
            measurements_cm, request.category, request.product, request.size
        )
        
        if "error" in analysis_result:
            raise HTTPException(status_code=404, detail=analysis_result["error"])
        
        # Convert analysis to response format
        analysis_details = {}
        for measurement, details in analysis_result["analysis"].items():
            analysis_details[measurement] = FitAnalysisDetail(
                score=details["score"],
                description=details["description"]
            )
        
        return FitAnalysisResponse(
            overall_rating=analysis_result["overall_rating"],
            overall_score=analysis_result["overall_score"],
            analysis=analysis_details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_fit: {e}")
        raise HTTPException(status_code=500, detail=f"Fit analysis failed: {str(e)}")

@router.get("/recommendations/stored", response_model=RecommendationResponse)
async def get_recommendations_for_stored_measurements(category: Optional[str] = None):
    """
    Get recommendations using stored measurements
    """
    try:
        # Load stored measurements
        measurements = measurement_service.load_measurements()
        
        if not measurements:
            raise HTTPException(status_code=404, detail="No measurements found. Please process images first.")
        
        # Create request with stored measurements
        user_measurements = UserMeasurements(**measurements)
        request = RecommendationRequest(measurements=user_measurements, category=category)
        
        # Use the existing recommendation endpoint
        return await get_recommendations(request)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_recommendations_for_stored_measurements: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.post("/best-fit/{product}", response_model=SizeRecommendation)
async def get_best_fit_for_product(product: str, measurements: UserMeasurements):
    """
    Get the best fitting size for a specific product
    """
    try:
        measurements_dict = measurements.dict()
        
        # Convert measurements to cm for analysis
        measurements_cm = measurement_service.convert_measurements_to_cm(measurements_dict)
        
        size, rating, score = recommendation_service.find_best_fit_for_product(
            measurements_cm, product
        )
        
        if not size:
            raise HTTPException(status_code=404, detail=f"No suitable fit found for product: {product}")
        
        return SizeRecommendation(
            size=size,
            rating=rating,
            score=score,
            product=product
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_best_fit_for_product: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find best fit: {str(e)}")

@router.post("/try-virtually-with-images", response_model=RecommendationResponse)
async def try_virtually_with_images(request: VirtualTryOnRequest):
    """
    Advanced virtual try-on endpoint that automatically processes user images
    
    This endpoint fetches user photos from their profile, processes them with U2Net + MediaPipe
    to extract real measurements, then provides size recommendations using the actual measurements.
    This gives the most accurate virtual try-on experience.
    """
    try:
        # For now, we'll expect the frontend to provide the image URLs directly
        # In a real implementation, you'd fetch from Supabase using the user_id
        # For this demo, we'll create a simplified version that expects image URLs in the request
        
        print(f"Processing virtual try-on for user: {request.user_id}")
        
        # This would be replaced with actual Supabase profile fetch
        # For now, return an error asking for image URLs to be provided directly
        raise HTTPException(
            status_code=400, 
            detail="This endpoint requires integration with Supabase. Please use try-virtually-with-image-urls endpoint instead."
        )
        
    except Exception as e:
        print(f"Error in try_virtually_with_images: {e}")
        raise HTTPException(status_code=500, detail=f"Virtual try-on with images failed: {str(e)}")

@router.post("/try-virtually-with-image-urls", response_model=RecommendationResponse) 
async def try_virtually_with_image_urls(request: ProcessImagesFromUrlsRequest):
    """
    Virtual try-on using image URLs for real measurement extraction
    
    Downloads images from URLs, processes them with U2Net + MediaPipe for real measurements,
    then provides accurate size recommendations.
    """
    try:
        print(f"Processing virtual try-on with image URLs")
        print(f"Front URL: {request.front_image_url}")
        print(f"Side URL: {request.side_image_url}")
        
        # Process images from URLs to get real measurements
        # Use height_cm if height is not provided
        height = request.height if request.height is not None else request.height_cm
        # Use weight_kg if weight is not provided
        weight = request.weight if request.weight is not None else request.weight_kg
        
        measurements = url_image_processor.process_images_from_urls(
            request.front_image_url,
            request.side_image_url,
            height,
            request.gender.upper(),
            weight,
            request.preferred_size,
            request.body_type
        )
        
        if not measurements:
            raise HTTPException(status_code=500, detail="Failed to extract measurements from images")
        
        print(f"Successfully extracted measurements: {measurements}")
        
        # Convert measurements to cm for analysis
        measurements_cm = measurement_service.convert_measurements_to_cm(measurements)
        
        # Default to T-shirt category if none specified
        category = "T-shirt"  # Fixed category for now
        
        # Use default size chart from Excel file (no custom size chart support in this simplified version)
        recommendations_data = recommendation_service.get_recommendations_for_category(
            measurements_cm, category
        )
        
        # Sort recommendations by score (highest first)
        sorted_recommendations = sorted(recommendations_data, key=lambda x: x.get("score", 0), reverse=True)
        
        # Convert to response format with enhanced descriptions for virtual try-on
        recommendations = []
        for i, rec in enumerate(sorted_recommendations[:3]):  # Limit to top 3 recommendations
            # Enhanced descriptions for virtual try-on
            if i == 0:  # Best fit
                if rec["score"] >= 9.0:
                    description = "Perfect fit based on your actual measurements! This size matches your body perfectly."
                elif rec["score"] >= 8.0:
                    description = "Excellent fit based on your measurements. This size offers optimal comfort and style."
                elif rec["score"] >= 7.0:
                    description = "Great fit based on your measurements. This size provides a comfortable and balanced look."
                else:
                    description = "Good fit based on your actual measurements."
            else:  # Alternative fits
                if rec["rating"] == "Tight":
                    description = f"Alternative option - may feel snug based on your measurements. Great if you prefer tighter-fitting clothes."
                elif rec["rating"] == "Loose":
                    description = f"Alternative option - offers a relaxed fit based on your measurements. Good for casual comfort."
                else:
                    description = f"Alternative sizing option based on your actual measurements."
            
            recommendations.append(SizeRecommendation(
                size=rec["size"],
                rating=rec["rating"],
                score=rec["score"],
                product=rec["product"]
            ))
        
        # Get the best recommended size
        best_recommendation = sorted_recommendations[0] if sorted_recommendations else None
        recommended_size = best_recommendation["size"] if best_recommendation else None
        fit_score = best_recommendation["score"] if best_recommendation else None
        
        # Create alternative sizes dictionary
        alternative_sizes = {}
        for rec in sorted_recommendations[1:]:
            alternative_sizes[rec["size"]] = rec["score"]
        
        return RecommendationResponse(
            categories=[category],
            products={category: recommendation_service.get_products_by_category(category)},
            recommendations=recommendations,
            recommended_size=recommended_size,
            fit_score=fit_score,
            alternative_sizes=alternative_sizes,
            measurements=measurements  # Include the extracted measurements
        )
        
    except Exception as e:
        print(f"Error in try_virtually_with_image_urls: {e}")
        raise HTTPException(status_code=500, detail=f"Virtual try-on with image URLs failed: {str(e)}")

@router.post("/try-virtually", response_model=RecommendationResponse)
async def try_virtually(request: RecommendationRequest):
    """
    Virtual try-on endpoint that provides comprehensive size recommendations
    
    This endpoint combines measurement analysis with fit recommendations specifically
    for virtual try-on functionality. It provides detailed size analysis,
    confidence scores, and fit descriptions optimized for the frontend virtual try-on experience.
    Can use custom size charts for specific products.
    """
    try:
        measurements_dict = request.measurements.dict()
        
        # Convert measurements to cm for analysis
        measurements_cm = measurement_service.convert_measurements_to_cm(measurements_dict)
        
        # Default to T-shirt category if none specified
        category = request.category or "T-shirt"
        
        # Check if custom size chart is provided
        if request.size_chart and len(request.size_chart) > 0:
            # Convert Pydantic models to dictionaries
            size_chart_data = [size_data.dict() for size_data in request.size_chart]
            
            # Use custom size chart for recommendations
            recommendations_data = recommendation_service.get_recommendations_with_custom_size_chart(
                measurements_cm, size_chart_data, category
            )
        else:
            # Use default size chart from Excel file
            recommendations_data = recommendation_service.get_recommendations_for_category(
                measurements_cm, category
            )
        
        # Sort recommendations by score (highest first)
        sorted_recommendations = sorted(recommendations_data, key=lambda x: x.get("score", 0), reverse=True)
        
        # Convert to response format with enhanced descriptions for virtual try-on
        recommendations = []
        for i, rec in enumerate(sorted_recommendations[:3]):  # Limit to top 3 recommendations
            # Enhanced descriptions for virtual try-on
            if i == 0:  # Best fit
                if rec["score"] >= 9.0:
                    description = "Perfect fit for virtual try-on! This size matches your measurements perfectly."
                elif rec["score"] >= 8.0:
                    description = "Excellent fit for virtual try-on. This size offers optimal comfort and style."
                elif rec["score"] >= 7.0:
                    description = "Great fit for virtual try-on. This size provides a comfortable and balanced look."
                else:
                    description = "Good fit for virtual try-on based on your measurements."
            else:  # Alternative fits
                if rec["rating"] == "Tight":
                    description = f"Alternative option - may feel snug. Great if you prefer tighter-fitting clothes."
                elif rec["rating"] == "Loose":
                    description = f"Alternative option - offers a relaxed fit. Good for casual comfort."
                else:
                    description = f"Alternative sizing option available for different fit preferences."
            
            recommendations.append(SizeRecommendation(
                size=rec["size"],
                rating=rec["rating"],
                score=rec["score"],
                product=rec["product"]
            ))
        
        return RecommendationResponse(
            categories=[category],
            products={category: recommendation_service.get_products_by_category(category) if not request.size_chart else ["CUSTOM_001"]},
            recommendations=recommendations
        )
        
    except Exception as e:
        print(f"Error in try_virtually: {e}")
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")


@router.get("/measurements", response_model=MeasurementsResponse)
async def get_measurements():
    """
    Get user body measurements
    
    Returns only the user's body measurements without size recommendations.
    This endpoint is designed for the initial dialog in the UI when users click "Compare My Size".
    """
    try:
        # Load stored measurements
        measurements = measurement_service.load_measurements()
        
        if not measurements:
            raise HTTPException(status_code=404, detail="No measurements found. Please process images first.")
        
        return MeasurementsResponse(
            measurements=measurements,
            message="Body measurements extracted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_measurements: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get measurements: {str(e)}")


@router.post("/size-recommendations", response_model=SizeRecommendationsResponse)
async def get_size_recommendations(request: SizeRecommendationsRequest):
    """
    Get size recommendations based on measurements and product
    
    Returns size recommendations without including the measurements in the response.
    This endpoint is designed for the second part of the UI where size recommendations are shown.
    """
    try:
        # Use the measurements provided in the request
        measurements = request.measurements
        
        # Get recommendations for the specified category
        recommendations_data = recommendation_service.get_recommendations_for_category(
            measurements, request.product_category
        )
        
        # Sort by score (highest first)
        sorted_recommendations = sorted(recommendations_data, key=lambda x: x.get("score", 0), reverse=True)
        
        # Get the best recommended size
        best_recommendation = sorted_recommendations[0] if sorted_recommendations else None
        recommended_size = best_recommendation["size"] if best_recommendation else None
        fit_score = best_recommendation["score"] if best_recommendation else None
        
        # Create alternative sizes dictionary
        alternative_sizes = {}
        for rec in sorted_recommendations[1:]:
            alternative_sizes[rec["size"]] = rec["score"]
        
        return SizeRecommendationsResponse(
            recommended_size=recommended_size,
            fit_score=fit_score,
            alternative_sizes=alternative_sizes,
            message="Size recommendations based on your measurements"
        )
        
    except Exception as e:
        print(f"Error in get_size_recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get size recommendations: {str(e)}")