"""
URL Image Processing Service
Downloads images from URLs and processes them with U2Net + MediaPipe
"""
import os
import tempfile
import requests
from typing import Dict, Any, Optional, Tuple
import uuid
from urllib.parse import urlparse

from .image_processing import image_service
from .measurement import measurement_service

class UrlImageProcessor:
    """Service for downloading and processing images from URLs"""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_image(self, image_url: str) -> Optional[str]:
        """
        Download an image from URL to temporary file
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Validate URL
            parsed = urlparse(image_url)
            if not parsed.scheme or not parsed.netloc:
                print(f"Invalid URL: {image_url}")
                return None
            
            # Generate unique filename
            file_extension = os.path.splitext(parsed.path)[1] or '.jpg'
            filename = f"downloaded_{uuid.uuid4().hex}{file_extension}"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Download image
            print(f"Downloading image from: {image_url}")
            response = self.session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                print(f"URL does not point to an image. Content-Type: {content_type}")
                return None
            
            # Save to file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded image to: {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {image_url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error downloading image: {e}")
            return None
    
    def process_images_from_urls(
        self, 
        front_image_url: str, 
        side_image_url: str,
        height: float,
        gender: str,
        weight: Optional[float] = None,
        preferred_size: Optional[str] = None,
        body_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Download and process images from URLs to extract measurements
        
        Args:
            front_image_url: URL to front view image
            side_image_url: URL to side view image
            height: Height in cm
            gender: Gender (M/F)
            weight: Weight in kg (optional)
            preferred_size: Preferred size (optional)
            body_type: Body type (optional)
            
        Returns:
            Dictionary containing measurements or None if failed
        """
        front_path = None
        side_path = None
        processed_front = None
        processed_side = None
        
        try:
            # Download images
            print("Downloading front image...")
            front_path = self.download_image(front_image_url)
            if not front_path:
                raise Exception("Failed to download front image")
            
            print("Downloading side image...")
            side_path = self.download_image(side_image_url)
            if not side_path:
                raise Exception("Failed to download side image")
            
            # Process images with U2Net background removal
            print("Processing images with U2Net and MediaPipe...")
            processed_front, processed_side = image_service.process_image_pair(front_path, side_path)
            
            if not processed_front or not processed_side:
                raise Exception(f"Image processing failed - Front: {'OK' if processed_front else 'FAILED'}, Side: {'OK' if processed_side else 'FAILED'}")
            
            # Extract measurements using MediaPipe
            print("Extracting measurements...")
            measurements = measurement_service.get_measurements_from_images(
                processed_front, processed_side, height, gender.upper(), weight, preferred_size, body_type
            )
            
            print("Successfully extracted measurements from URLs")
            return measurements
            
        except Exception as e:
            print(f"Error processing images from URLs: {e}")
            return None
        
        finally:
            # Cleanup downloaded files
            for path in [front_path, side_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            
            # Keep processed images for potential future use
            # They will be cleaned up by the regular cleanup process

# Global service instance
url_image_processor = UrlImageProcessor()