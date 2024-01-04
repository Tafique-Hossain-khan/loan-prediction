import os
from src.exception import CustomException
from src.logger import logging
import sys
import numpy as np
import pandas as pd
import pickle 


def save_object(file_path,obj):

    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'wb') as f:
            pickle.dump(obj,f)

    except CustomException as e:
        raise CustomException(e,sys)
    

def load_object(file_path):

    try:
        with open(file_path,'rb') as f:
            pickle.loads(f)

            
    except CustomException as e:
        raise CustomException(e,sys)

    