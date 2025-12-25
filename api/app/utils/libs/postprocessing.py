"""
Name: Post-processing class file
Description: This file contains post-processing classes.
Version: [release][3.2]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Anodev (OPHoperHPO)[https://github.com/OPHoperHPO] .
License: Apache License 2.0
"""
import logging
from .strings import POSTPROCESS_METHODS

logger = logging.getLogger(__name__)


def method_detect(method: str):
    """Detects which method to use and returns its object"""
    if method in POSTPROCESS_METHODS:
        # For standalone version, we'll return None for all postprocessing methods
        # to use the standard background removal
        return None
    else:
        return False