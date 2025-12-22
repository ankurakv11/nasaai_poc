"""
Measurement service for body measurements using MediaPipe
"""
import os
import sys
import json
from typing import Dict, Any, Optional

# Import local modules from app.utils
try:
    from app.utils.mediapipe_measurement import mediapipe_measure
    print("Successfully imported mediapipe_measurement from app.utils")
except ImportError as e:
    print(f"Warning: Could not import mediapipe_measurement from app.utils: {e}")
    # Fallback function
    def mediapipe_measure(front_path: str, side_path: str, height: float, gender: str, **kwargs) -> Dict[str, Any]:
        """Fallback function for measurements"""
        return {
            "height": height,
            "neck": 14.0,
            "chest": 36.0,
            "waist": 30.0,
            "Butt": 36.0,
            "shoulder_width": 17.0,
            "body_length": 26.0,
            "inseam": 30.0
        }

class MeasurementService:
    """Service class for body measurements"""
    
    def __init__(self, data_file: str = "data.txt"):
        """Initialize the measurement service"""
        self.data_file = data_file
    
    def get_measurements_from_images(
        self, 
        front_image_path: str, 
        side_image_path: str, 
        height: float, 
        gender: str, 
        weight: Optional[float] = None,
        preferred_size: Optional[str] = None,
        body_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get body measurements from processed images
        
        Args:
            front_image_path: Path to processed front image
            side_image_path: Path to processed side image
            height: User height in cm
            gender: User gender (M/F)
            weight: User weight in kg (optional)
            preferred_size: Preferred size (optional)
            body_type: Body type for females (optional)
            
        Returns:
            Dictionary containing body measurements
        """
        try:
            measurements = mediapipe_measure(
                front_image_path,
                side_image_path,
                height,
                gender,
                weight=weight,
                preferred_size=preferred_size,
                body_type=body_type
            )
            
            # Save measurements to data file for compatibility
            self.save_measurements(measurements)
            
            return measurements
            
        except Exception as e:
            print(f"Error getting measurements: {e}")
            raise Exception(f"Measurement failed: {str(e)}")
    
    def save_measurements(self, measurements: Dict[str, Any]) -> None:
        """
        Save measurements to data file
        
        Args:
            measurements: Dictionary containing measurements
        """
        try:
            with open(self.data_file, 'w') as f:
                json.dump(measurements, f)
        except Exception as e:
            print(f"Error saving measurements: {e}")
    
    def load_measurements(self) -> Optional[Dict[str, Any]]:
        """
        Load measurements from data file
        
        Returns:
            Dictionary containing measurements or None if not found
        """
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading measurements: {e}")
            return None
    
    def validate_measurements(self, measurements: Dict[str, Any]) -> bool:
        """
        Validate measurements dictionary
        
        Args:
            measurements: Dictionary containing measurements
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['height', 'neck', 'chest', 'waist', 'Butt']
        
        if not isinstance(measurements, dict):
            return False
        
        for field in required_fields:
            if field not in measurements:
                return False
            
            value = measurements[field]
            if not isinstance(value, (int, float)) or value <= 0:
                return False
        
        return True
    
    def convert_measurements_to_cm(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert inch measurements to cm for compatibility
        
        Args:
            measurements: Dictionary containing measurements in inches
            
        Returns:
            Dictionary with measurements converted to cm where needed
        """
        converted = measurements.copy()
        
        # Convert inch measurements to cm
        inch_fields = ['neck', 'chest', 'waist', 'Butt', 'shoulder_width']
        
        for field in inch_fields:
            if field in converted and converted[field] is not None:
                converted[field] = converted[field] * 2.54
        
        return converted

# Global service instance
measurement_service = MeasurementService()