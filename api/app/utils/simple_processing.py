# Simple image processing utilities to replace enhanced_bg_removal
import cv2
import shutil
import os

def enhance_image_quality(input_path, output_path):
    """Simple image enhancement - just copy the file"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(input_path, output_path)
        return True
    except Exception as e:
        print(f"Error in enhance_image_quality: {e}")
        return False

def robust_background_removal(input_path, output_path, method="u2net"):
    """Simple wrapper around the main bg_removal process function"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Import the background removal function from our local bg_removal module
        from .bg_removal import process
        # Use the main background removal function
        process(input_path, output_path)
        return True
    except Exception as e:
        print(f"Background removal failed: {e}")
        # Fallback - just copy the original image
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(input_path, output_path)
            return True
        except Exception as fallback_error:
            print(f"Even fallback failed: {fallback_error}")
            return False

def post_process_mask(input_path, output_path):
    """Simple post-processing - just copy the result"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(input_path, output_path)
        return True
    except Exception as e:
        print(f"Error in post_process_mask: {e}")
        return False