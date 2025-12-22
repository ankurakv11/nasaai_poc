# #############################################################################
# #############################################################################
# Concept by	    :	Ankit
# Designed by 	    :	Shobhit
# Coded By			:	Shobhit
# Tested By		    :	Shobhit

# #############################################################################
# OTHER INFORMATION
# Developer			: 	Shobhit
# Year				: 	2020
# Version			:   1.0
# Library			:   bg_removal
# Country			: 	India
# Duration			: 	01.12.2020- 14.12.2020
# #############################################################################
# #############################################################################

# #############################################################################
# PURPOSE:	IMPORT MODULES
# #############################################################################
import os
import tqdm
import logging
from .libs.networks import model_detect
from .libs import preprocessing
from .libs import postprocessing

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def get_cached_model(model_name="u2net"):
    """Get cached model from main app if available"""
    try:
        # Import here to avoid circular imports
        from main import get_u2net_model
        cached_model = get_u2net_model()
        if cached_model and model_name == "u2net":
            logger.info("Using cached U2Net model")
            return cached_model
    except ImportError:
        logger.debug("Could not import cached model from main app")
    except Exception as e:
        logger.debug(f"Error getting cached model: {e}")
    return None

def __work_mode__(path: str):
    if os.path.isfile(path):  # Input is file
        return "file"
    if os.path.isdir(path):  # Input is dir
        return "dir"
    else:
        return "no"

def __save_image_file__(img, file_name, output_path, wmode):
    # create output directory if it doesn't exist
    folder = os.path.dirname(output_path)
    if folder != '':
        os.makedirs(folder, exist_ok=True)
    if wmode == "file":
        file_name_out = os.path.basename(output_path)
        if file_name_out == '':
            # Change file extension to png
            file_name = os.path.splitext(file_name)[0] + '.png'
            # Save image
            img.save(os.path.join(output_path, file_name))
        else:
            try:
                # Save image
                img.save(output_path)
            except OSError as e:
                if str(e) == "cannot write mode RGBA as JPEG":
                    raise OSError("Error! "
                                  "Please indicate the correct extension of the final file, for example: .png")
                else:
                    raise e
    else:
        # Change file extension to png
        file_name = os.path.splitext(file_name)[0] + '.png'
        # Save image
        img.save(os.path.join(output_path, file_name))


def process(input_path, output_path, model_name="u2net",
            preprocessing_method_name="None", postprocessing_method_name="No"):

    if input_path is None or output_path is None:
        raise Exception("Bad parameters! Please specify input path and output path.")

    # Try to get cached model first
    model = get_cached_model(model_name)
    if not model:
        # Fallback to loading new model if cached one not available
        model = model_detect(model_name)  # Load model
        if not model:
            logger.warning("Warning! You specified an invalid model type. "
                           "For image processing, the model with the best processing quality will be used. "
                           "(u2net)")
            model_name = "u2net"  # If the model line is wrong, select the model with better quality.
            model = model_detect(model_name)  # Load model
    preprocessing_method = preprocessing.method_detect(preprocessing_method_name)
    postprocessing_method = postprocessing.method_detect(postprocessing_method_name)
    wmode = __work_mode__(input_path)  # Get work mode
    if wmode == "file":  # File work mode
        image = model.process_image(input_path, preprocessing_method, postprocessing_method)
        __save_image_file__(image, os.path.basename(input_path), output_path, wmode)
    elif wmode == "dir":  # Dir work mode
        # Start process
        files = os.listdir(input_path)
        for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
            file_path = os.path.join(input_path, file)
            image = model.process_image(file_path, preprocessing_method, postprocessing_method)
            __save_image_file__(image, file, output_path, wmode)
    else:
        raise Exception("Bad input parameter! Please indicate the correct path to the file or folder.")