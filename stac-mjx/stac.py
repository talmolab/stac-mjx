"""this file will contain the STAC class, which essentially implements the logic inside stac_test.py
so this will be the interface that people actually interface with to use stac
"""
    
import mujoco
from typing import Text

import utils

class STAC:
    def __init__(
        self,
        param_path: Text,
    ):  
        # Still feels weird to use a global params variable...
        utils.init_params(param_path)
        