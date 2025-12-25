"""
Size recommendation service
Provides size recommendations based on user measurements
"""
import os
import json
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class SizeRecommendationService:
    """
    Service for providing size recommendations based on user measurements
    """
    
    def __init__(self):
        """
        Initialize the service with size charts
        """
        # Load size charts from data files
        self.size_charts = {}
        self.load_size_charts()
        
        # Define measurement importance weights for different categories
        self.importance_weights = {
            "Tops": {
                "chest": 0.40,
                "shoulder": 0.30,
                "waist": 0.20,
                "neck": 0.10
            },
            "Bottoms": {
                "waist": 0.45,
                "hip": 0.35,
                "inseam": 0.20
            },
            "Dresses": {
                "chest": 0.30,
                "waist": 0.30,
                "hip": 0.25,
                "shoulder": 0.15
            },
            "T-shirt": {  # Added for simplified demo
                "chest": 0.45,
                "shoulder": 0.30,
                "waist": 0.25
            }
        }
    
    def load_size_charts(self):
        """
        Load size charts from data files
        """
        # For demonstration, we'll use hardcoded size charts
        # In a real implementation, these would be loaded from files
        
        # Example T-shirt size chart (unisex)
        self.size_charts["T-shirt"] = {
            "XS": {
                "chest_inches": 34,
                "waist_inches": 28,
                "shoulder_inches": 15
            },
            "S": {
                "chest_inches": 36,
                "waist_inches": 30,
                "shoulder_inches": 16
            },
            "M": {
                "chest_inches": 38,
                "waist_inches": 32,
                "shoulder_inches": 17
            },
            "L": {
                "chest_inches": 40,
                "waist_inches": 34,
                "shoulder_inches": 18
            },
            "XL": {
                "chest_inches": 42,
                "waist_inches": 36,
                "shoulder_inches": 19
            },
            "XXL": {
                "chest_inches": 44,
                "waist_inches": 38,
                "shoulder_inches": 20
            }
        }
        
        # Add more size charts for other categories as needed
    
    def get_categories(self) -> List[str]:
        """
        Get all available garment categories
        """
        return list(self.importance_weights.keys())
    
    def get_products_by_category(self, category: str) -> List[str]:
        """
        Get all products for a specific category
        """
        # For demonstration, we'll return a list of example products
        if category == "Tops":
            return ["T-shirt", "Shirt", "Sweater"]
        elif category == "Bottoms":
            return ["Jeans", "Chinos", "Shorts"]
        elif category == "Dresses":
            return ["Maxi Dress", "Mini Dress", "Midi Dress"]
        elif category == "T-shirt":
            return ["Classic Tee", "V-neck Tee", "Graphic Tee"]
        else:
            return []
    
    def get_sizes_for_product(self, product: str) -> List[str]:
        """
        Get all available sizes for a specific product
        """
        # For demonstration, we'll return a standard set of sizes
        if product in ["T-shirt", "Shirt", "Sweater", "Classic Tee", "V-neck Tee", "Graphic Tee"]:
            return ["XS", "S", "M", "L", "XL", "XXL"]
        elif product in ["Jeans", "Chinos", "Shorts"]:
            return ["28", "30", "32", "34", "36", "38", "40"]
        elif product in ["Maxi Dress", "Mini Dress", "Midi Dress"]:
            return ["XS", "S", "M", "L", "XL"]
        else:
            return []
    
    def _calculate_fit_score(self, user_measurements: Dict, size_measurements: Dict, category: str) -> Tuple[float, str]:
        """
        Calculate fit score for a specific size
        
        Args:
            user_measurements: User's measurements
            size_measurements: Size measurements
            category: Product category
            
        Returns:
            Tuple of (score, rating)
        """
        # Get importance weights for the category
        weights = self.importance_weights.get(category, {})
        if not weights:
            # Use default weights if category not found
            weights = {"chest": 0.4, "waist": 0.3, "hip": 0.2, "shoulder": 0.1}
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        
        for measurement, weight in weights.items():
            # Skip if measurement not available in either user or size measurements
            user_key = measurement
            size_key = f"{measurement}_inches"
            
            if user_key not in user_measurements or size_key not in size_measurements:
                continue
            
            user_value = user_measurements[user_key]
            size_value = size_measurements[size_key]
            
            # Calculate difference as percentage of size measurement
            if size_value == 0:
                continue
                
            difference = abs(user_value - size_value) / size_value
            
            # Convert to score (0-10 scale, 10 being perfect fit)
            # Smaller difference = higher score
            measurement_score = 10 - min(difference * 20, 10)  # Cap at 0
            
            weighted_score += measurement_score * weight
            total_weight += weight
        
        # Calculate final score
        if total_weight == 0:
            return 0, "Unknown"
            
        final_score = weighted_score / total_weight
        
        # Determine rating based on score
        if final_score >= 8.5:
            rating = "Perfect Fit"
        elif final_score >= 7.5:
            rating = "Good Fit"
        elif final_score >= 6.0:
            rating = "Acceptable Fit"
        elif final_score >= 4.5:
            rating = "Tight Fit" if self._is_tight_fit(user_measurements, size_measurements) else "Loose Fit"
        else:
            rating = "Too Tight Fit" if self._is_tight_fit(user_measurements, size_measurements) else "Too Loose Fit"
        
        return final_score, rating
    
    def _is_tight_fit(self, user_measurements: Dict, size_measurements: Dict) -> bool:
        """
        Determine if the fit is tight or loose
        
        Args:
            user_measurements: User's measurements
            size_measurements: Size measurements
            
        Returns:
            True if tight fit, False if loose fit
        """
        # Count measurements where user value is larger than size value
        tight_count = 0
        loose_count = 0
        
        for measurement in ["chest", "waist", "hip", "shoulder"]:
            user_key = measurement
            size_key = f"{measurement}_inches"
            
            if user_key not in user_measurements or size_key not in size_measurements:
                continue
                
            user_value = user_measurements[user_key]
            size_value = size_measurements[size_key]
            
            if user_value > size_value:
                tight_count += 1
            elif user_value < size_value:
                loose_count += 1
        
        # If more measurements are tight than loose, it's a tight fit
        return tight_count > loose_count
    
    def get_recommendations_with_custom_size_chart(self, user_measurements: Dict, size_chart: List[Dict], category: str) -> List[Dict]:
        """
        Get size recommendations using a custom size chart
        
        Args:
            user_measurements: User's measurements
            size_chart: Custom size chart data
            category: Product category
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Ensure user measurements are in inches
        # This assumes the frontend is sending measurements in inches
        # If not, you would need to convert them here
        
        # Process each size in the custom size chart
        for size_data in size_chart:
            size_label = size_data.get("size_label", "")
            if not size_label:
                continue
                
            # Calculate fit score
            score, rating = self._calculate_fit_score(user_measurements, size_data, category)
            
            recommendations.append({
                "size": size_label,
                    "rating": rating,
                "score": score,
                "product": "CUSTOM_001"  # Custom product ID for custom size charts
                })
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations
    
    def get_recommendations_for_category(self, user_measurements: Dict, category: str) -> List[Dict]:
        """
        Get size recommendations for a specific category
        
        Args:
            user_measurements: User's measurements
            category: Product category
            
        Returns:
            List of recommendations
        """
        # For demonstration, we'll use the T-shirt size chart for all categories
        size_chart = self.size_charts.get(category, self.size_charts.get("T-shirt", {}))
        
        recommendations = []
        
        for size, size_data in size_chart.items():
            # Calculate fit score
            score, rating = self._calculate_fit_score(user_measurements, size_data, category)
            
            # Add to recommendations
            recommendations.append({
                "size": size,
                    "rating": rating,
                    "score": score,
                "product": f"{category}_001"  # Example product ID
                })
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations
    
    def get_all_recommendations(self, user_measurements: Dict) -> Dict:
        """
        Get recommendations for all categories
        
        Args:
            user_measurements: User's measurements
            
        Returns:
            Dictionary with recommendations, categories, and products
        """
        all_recommendations = []
        categories = self.get_categories()
        products = {}
        
        for category in categories:
            category_recommendations = self.get_recommendations_for_category(user_measurements, category)
            all_recommendations.extend(category_recommendations)
            
            # Get products for this category
            products[category] = self.get_products_by_category(category)
        
        # Sort by score (highest first)
        all_recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "recommendations": all_recommendations,
            "categories": categories,
            "products": products
        }
    
    def analyze_fit(self, user_measurements: Dict, category: str, product: str, size: str) -> Dict:
        """
        Analyze fit for a specific garment
        
        Args:
            user_measurements: User's measurements
            category: Product category
            product: Product ID
            size: Size ID
            
        Returns:
            Dictionary with fit analysis
        """
        # Get size chart for the category
        size_chart = self.size_charts.get(category, {})
        
        # Get size data
        size_data = size_chart.get(size, {})
        if not size_data:
            return {"error": f"Size {size} not found for {category}"}
        
        # Get importance weights for the category
        weights = self.importance_weights.get(category, {})
        if not weights:
            weights = {"chest": 0.4, "waist": 0.3, "hip": 0.2, "shoulder": 0.1}
        
        # Analyze each measurement
        analysis = {}
        total_weight = 0
        weighted_score = 0
        
        for measurement, weight in weights.items():
            # Skip if measurement not available in either user or size measurements
            user_key = measurement
            size_key = f"{measurement}_inches"
            
            if user_key not in user_measurements or size_key not in size_data:
                continue
            
            user_value = user_measurements[user_key]
            size_value = size_data[size_key]
            
            # Calculate difference as percentage of size measurement
            if size_value == 0:
                continue
                
            difference = (user_value - size_value) / size_value
            
            # Convert to score (0-10 scale, 10 being perfect fit)
            # Smaller difference = higher score
            measurement_score = 10 - min(abs(difference) * 20, 10)  # Cap at 0
            
            # Determine description based on difference
            if abs(difference) < 0.05:
                description = "Perfect fit"
            elif abs(difference) < 0.10:
                description = "Good fit"
            elif abs(difference) < 0.15:
                description = "Acceptable fit"
            else:
                if difference > 0:
                    description = "Too tight"
                else:
                    description = "Too loose"
            
            analysis[measurement] = {
                "score": measurement_score,
                "description": description
            }
            
            weighted_score += measurement_score * weight
            total_weight += weight
        
        # Calculate overall score and rating
        if total_weight == 0:
            overall_score = 0
            overall_rating = "Unknown"
        else:
            overall_score = weighted_score / total_weight
            
            # Determine overall rating
            if overall_score >= 8.5:
                overall_rating = "Perfect Fit"
            elif overall_score >= 7.5:
                overall_rating = "Good Fit"
            elif overall_score >= 6.0:
                overall_rating = "Acceptable Fit"
            elif overall_score >= 4.5:
                overall_rating = "Tight Fit" if self._is_tight_fit(user_measurements, size_data) else "Loose Fit"
            else:
                overall_rating = "Too Tight Fit" if self._is_tight_fit(user_measurements, size_data) else "Too Loose Fit"
        
        return {
            "overall_score": overall_score,
            "overall_rating": overall_rating,
            "analysis": analysis
        }
    
    def find_best_fit_for_product(self, user_measurements: Dict, product: str) -> Tuple[str, str, float]:
        """
        Find the best fitting size for a specific product
        
        Args:
            user_measurements: User's measurements
            product: Product ID
            
        Returns:
            Tuple of (size, rating, score)
        """
        # For demonstration, we'll map product to category
        # In a real implementation, you would look up the product in a database
        category = "T-shirt"  # Default
        
        if product.startswith("T-shirt"):
            category = "T-shirt"
        elif product.startswith("Jeans") or product.startswith("Chinos") or product.startswith("Shorts"):
            category = "Bottoms"
        elif product.startswith("Dress"):
            category = "Dresses"
        
        # Get recommendations for the category
        recommendations = self.get_recommendations_for_category(user_measurements, category)
        
        # Return the best recommendation
        if recommendations:
            best = recommendations[0]
            return best["size"], best["rating"], best["score"]
        else:
            return "", "Unknown", 0.0

# Global service instance
recommendation_service = SizeRecommendationService()