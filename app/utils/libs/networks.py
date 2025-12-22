"""
Name: Neural networks file.
Description: This file contains neural network classes.
Version: [release][3.2]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Anodev (OPHoperHPO)[https://github.com/OPHoperHPO] .
License: Apache License 2.0
"""
import logging
import os
import time

import numpy as np
from PIL import Image
from skimage import io, transform
try:
    import scipy.ndimage
    from skimage import morphology
    HAS_ADVANCED_PROCESSING = True
except ImportError:
    HAS_ADVANCED_PROCESSING = False

from . import strings

logger = logging.getLogger(__name__)


def model_detect(model_name):
    """Detects which model to use and returns its object"""
    models_names = strings.MODELS_NAMES
    if model_name in models_names:
        if model_name == "xception_model" or model_name == "mobile_net_model":
            return TFSegmentation(model_name)
        elif "u2net" in model_name:
            return U2NET(model_name)
        elif "basnet" == model_name:
            return BasNet(model_name)
        else:
            return False
    else:
        return False


class U2NET:
    """U^2-Net model interface"""

    def __init__(self, name="u2net"):
        import torch
        from torch.autograd import Variable
        try:
            from .u2net import U2NET as U2NET_DEEP
            from .u2net import U2NETP as U2NETP_DEEP
        except ImportError:
            # Try absolute import
            from app.utils.libs.u2net import U2NET as U2NET_DEEP
            from app.utils.libs.u2net import U2NETP as U2NETP_DEEP
        
        self.Variable = Variable
        self.torch = torch
        self.U2NET_DEEP = U2NET_DEEP
        self.U2NETP_DEEP = U2NETP_DEEP

        if name == 'u2net':  # Load model
            logger.debug("Loading a U2NET model (176.6 mb) with better quality but slower processing.")
            net = self.U2NET_DEEP()
        elif name == 'u2netp':
            logger.debug("Loading a U2NETp model (4 mb) with lower quality but fast processing.")
            net = self.U2NETP_DEEP()
        else:
            raise Exception("Unknown u2net model!")
            
        # Load pre-trained weights from app/models directory
        model_dir = os.path.join("app", "models", "u2net")
        if name == 'u2net':
            model_path = os.path.join(model_dir, "u2net.pth")
        else:
            model_path = os.path.join(model_dir, "u2netp.pth")
        
        # Try to load the weights
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading pre-trained weights from: {model_path}")
                if self.torch.cuda.is_available():
                    net.load_state_dict(self.torch.load(model_path))
                    net.cuda()
                    logger.info("Model loaded successfully on GPU")
                else:
                    net.load_state_dict(self.torch.load(model_path, map_location='cpu'))
                    logger.info("Model loaded successfully on CPU")
            except Exception as e:
                logger.warning(f"Failed to load model weights: {e}. Using random initialization.")
        else:
            logger.warning(f"Pre-trained model weights not found at {model_path}. Using random initialization.")
        
        net.eval()
        self.__net__ = net  # Define model

    def process_image(self, data, preprocessing=None, postprocessing=None):
        """
        Removes background from image and returns PIL RGBA Image.
        """
        if isinstance(data, str):
            logger.debug("Load image: {}".format(data))
        image, org_image = self.__load_image__(data)  # Load image
        if image is False or org_image is False:
            return False

        try:
            if preprocessing:
                image = preprocessing.run(self, image, org_image)
            else:
                image = self.__get_output__(image, org_image)
            if postprocessing:
                image = postprocessing.run(self, image, org_image)
            return image
        finally:
            # Ensure GPU memory is freed even if an error occurs
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

    def __get_output__(self, image, org_image):
        """Returns output from a neural network"""
        start_time = time.time()
        image = image.type(self.torch.FloatTensor)
        if self.torch.cuda.is_available():
            image = self.Variable(image.cuda())
        else:
            image = self.Variable(image)

        try:
            mask, d2, d3, d4, d5, d6, d7 = self.__net__(image)
            logger.debug("Mask prediction completed")

            # Free intermediate tensors from GPU memory
            del d2, d3, d4, d5, d6, d7
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

            mask = mask[:, 0, :, :]
            mask = self.__normalize__(mask)
            mask = self.__prepare_mask__(mask, org_image.size)

            # Free GPU memory after processing
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
                logger.debug("GPU cache cleared after inference")
        except Exception as e:
            logger.warning(f"Neural network processing failed: {e}. Using fallback simple mask.")
            # Create a simple fallback mask (center region)
            mask = self.__create_fallback_mask__(org_image.size)
            # Clear cache even on error
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

        empty = Image.new("RGBA", org_image.size)
        image = Image.composite(org_image, empty, mask)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __create_fallback_mask__(self, image_size):
        """Create a simple fallback mask when neural network fails"""
        width, height = image_size
        mask = Image.new("L", image_size, 0)
        
        # Create a simple center oval mask
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        
        # Define oval bounds (80% of image centered)
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)
        bbox = [margin_x, margin_y, width - margin_x, height - margin_y]
        
        # Draw filled oval
        draw.ellipse(bbox, fill=255)
        
        logger.debug("Created fallback mask")
        return mask

    def __load_image__(self, data):
        """Loads an image file for processing"""
        if isinstance(data, str):
            try:
                pil_image = Image.open(data)
                
                # Apply EXIF orientation if present
                try:
                    if hasattr(pil_image, '_getexif'):
                        exif = pil_image._getexif()
                        if exif is not None:
                            orientation = exif.get(274)
                            if orientation == 3:
                                pil_image = pil_image.rotate(180, expand=True)
                            elif orientation == 6:
                                pil_image = pil_image.rotate(270, expand=True)
                            elif orientation == 8:
                                pil_image = pil_image.rotate(90, expand=True)
                except (AttributeError, KeyError, TypeError):
                    pass
                
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                image = np.array(pil_image)
                
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data)
                return False, False
        else:
            image = np.array(data)
            pil_image = data
        
        # Calculate optimal size for processing
        original_height, original_width = image.shape[:2]
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:
            target_width = max(512, min(640, original_width))
            target_height = int(target_width / aspect_ratio)
        else:
            target_height = max(512, min(640, original_height))
            target_width = int(target_height * aspect_ratio)
        
        # Ensure dimensions are multiples of 32 for U2NET
        target_width = ((target_width + 31) // 32) * 32
        target_height = ((target_height + 31) // 32) * 32
        
        image = transform.resize(image, (target_height, target_width), 
                               mode='constant', anti_aliasing=True, preserve_range=True)
        
        image = self.__ndrarray2tensor__(image)
        return image, pil_image

    def __ndrarray2tensor__(self, image: np.ndarray):
        """Converts a NumPy array to a tensor"""
        tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
        image /= np.max(image)
        if image.shape[2] == 1:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = np.expand_dims(tmp_img, 0)
        return self.torch.from_numpy(tmp_img)

    def __normalize__(self, predicted):
        """Normalize the predicted map"""
        ma = self.torch.max(predicted)
        mi = self.torch.min(predicted)
        out = (predicted - mi) / (ma - mi)
        return out

    @staticmethod
    def __prepare_mask__(predict, image_size):
        """Simplified mask preparation"""
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        
        threshold = 0.5
        mask = np.where(predict_np > threshold, 1.0, 0.0)
        
        if HAS_ADVANCED_PROCESSING:
            try:
                mask = scipy.ndimage.gaussian_filter(mask, sigma=0.5)
                binary_mask = mask > 0.5
                binary_mask = scipy.ndimage.binary_fill_holes(binary_mask)
                binary_mask = morphology.remove_small_objects(binary_mask, min_size=100)
                mask = binary_mask.astype(np.float32)
            except Exception:
                pass
        
        mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        mask = mask.resize(image_size, resample=Image.LANCZOS)
        return mask


# Simplified versions of other classes for compatibility
class BasNet:
    """Simplified BasNet implementation"""
    def __init__(self, name="basnet"):
        raise NotImplementedError("BasNet model not implemented in standalone version")

class TFSegmentation:
    """Simplified TensorFlow implementation"""
    def __init__(self, model_type):
        raise NotImplementedError("TensorFlow models not implemented in standalone version")