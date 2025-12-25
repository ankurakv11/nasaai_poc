"""
Image processing service for background removal and enhancement
"""
import os
import sys
import tempfile
from typing import Tuple, Optional

# Import local modules from app.utils
try:
    from app.utils.simple_processing import enhance_image_quality, robust_background_removal, post_process_mask
    print("Successfully imported simple_processing modules from app.utils")
except ImportError as e:
    print(f"Warning: Could not import image processing modules from app.utils: {e}")
    
    # Fallback functions
    def enhance_image_quality(input_path: str, output_path: str):
        """Fallback function for image enhancement"""
        import shutil
        print(f"Fallback: Copying {input_path} to {output_path}")
        shutil.copy2(input_path, output_path)
    
    def robust_background_removal(input_path: str, output_path: str, method: str = 'hybrid'):
        """Fallback function for background removal"""
        import shutil
        print(f"Fallback: Copying {input_path} to {output_path} (no background removal)")
        shutil.copy2(input_path, output_path)
    
    def post_process_mask(input_path: str, output_path: str):
        """Fallback function for post-processing"""
        import shutil
        print(f"Fallback: Copying {input_path} to {output_path}")
        shutil.copy2(input_path, output_path)

class ImageProcessingService:
    """Service class for image processing operations"""
    
    def __init__(self, upload_dir: str = "uploads"):
        """Initialize the image processing service"""
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create subdirectories
        self.enhanced_dir = os.path.join(upload_dir, "enhanced")
        self.processed_dir = os.path.join(upload_dir, "processed")
        self.temp_dir = os.path.join(upload_dir, "temp")
        
        for directory in [self.enhanced_dir, self.processed_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def process_single_image(self, image_path: str, base_name: str) -> Optional[str]:
        """
        Process a single image through the enhancement and background removal pipeline
        
        Args:
            image_path: Path to the input image
            base_name: Base name for output files
            
        Returns:
            Path to the processed image or None if processing fails
        """
        try:
            print(f"Starting to process {base_name} from {image_path}")
            
            # Check if input file exists
            if not os.path.exists(image_path):
                print(f"Error: Input file does not exist: {image_path}")
                return None
            
            # Define file paths
            enhanced_path = os.path.join(self.enhanced_dir, f"{base_name}_enhanced.jpg")
            temp_path = os.path.join(self.temp_dir, f"{base_name}_temp.png")
            final_path = os.path.join(self.processed_dir, f"{base_name}_processed.png")
            
            print(f"Processing {base_name}: {image_path} -> {final_path}")
            
            # Processing pipeline with individual error handling
            try:
                print(f"Step 1: Enhancing image quality...")
                enhance_image_quality(image_path, enhanced_path)
                if not os.path.exists(enhanced_path):
                    print(f"Warning: Enhanced image not created, using original")
                    enhanced_path = image_path
            except Exception as e:
                print(f"Error in enhance_image_quality: {e}")
                print(f"Warning: Enhancement failed: {e}, using original image")
                enhanced_path = image_path
            
            try:
                print(f"Step 2: Background removal...")
                robust_background_removal(enhanced_path, temp_path, method='hybrid')
                if not os.path.exists(temp_path):
                    print(f"Warning: Background removal failed, copying original")
                    import shutil
                    # Ensure the temp directory exists
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    shutil.copy2(enhanced_path, temp_path)
            except Exception as e:
                print(f"Warning: Background removal failed: {e}, copying original")
                import shutil
                # Ensure the temp directory exists
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                shutil.copy2(enhanced_path, temp_path)
            
            try:
                print(f"Step 3: Post-processing...")
                post_process_mask(temp_path, final_path)
                if not os.path.exists(final_path):
                    print(f"Warning: Post-processing failed, copying temp file")
                    import shutil
                    # Ensure the processed directory exists
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    shutil.copy2(temp_path, final_path)
            except Exception as e:
                print(f"Warning: Post-processing failed: {e}, copying temp file")
                import shutil
                # Ensure the processed directory exists
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.copy2(temp_path, final_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path) and temp_path != final_path:
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            if os.path.exists(final_path):
                print(f"Successfully processed {base_name}")
                return final_path
            else:
                print(f"Final processed file not created for {base_name}")
                return None
            
        except Exception as e:
            print(f"Error processing image {base_name}: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def process_image_pair(self, front_image_path: str, side_image_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process a pair of images (front and side)
        
        Args:
            front_image_path: Path to the front image
            side_image_path: Path to the side image
            
        Returns:
            Tuple of (processed_front_path, processed_side_path)
        """
        front_processed = self.process_single_image(front_image_path, "front")
        side_processed = self.process_single_image(side_image_path, "side")
        
        return front_processed, side_processed
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file content to disk
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Returns:
            Path to the saved file
        """
        try:
            print(f"Saving uploaded file: {filename} ({len(file_content)} bytes)")
            
            # Generate safe filename
            safe_filename = os.path.basename(filename)
            file_path = os.path.join(self.upload_dir, safe_filename)
            
            # Ensure unique filename
            counter = 1
            base_name, ext = os.path.splitext(safe_filename)
            while os.path.exists(file_path):
                file_path = os.path.join(self.upload_dir, f"{base_name}_{counter}{ext}")
                counter += 1
            
            print(f"Saving to: {file_path}")
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            if os.path.exists(file_path):
                print(f"Successfully saved file: {file_path}")
                return file_path
            else:
                print(f"Error: File was not saved: {file_path}")
                return None
                
        except Exception as e:
            print(f"Error saving uploaded file {filename}: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir, exist_ok=True)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

# Global service instance
image_service = ImageProcessingService()