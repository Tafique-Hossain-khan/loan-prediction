import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import os
import sys
from sklearn.ensemble import RandomForestClassifier 
from src.utils import save_object

from sklearn.model_selection import cross_val_score
class ModelTranerConfig:
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')


class ModelTraner:


    def __init__(self) -> None:
        self.model_traner_config = ModelTranerConfig()


    
    def train_model(self,train_arr,test_arr):
        try:

            logging.info("spliting the data")
            X = train_arr[:,:-1]
            y = train_arr[:,-1]

            rf = RandomForestClassifier()
            rf.fit(X,y)

            score = cross_val_score(rf,X,y,cv=5)
            logging.info(f"The accuarcy of the model is:{score.mean()}")

            save_object(self.model_traner_config.trained_model_file_path,rf)
            logging.info("Traning complited!")

        except CustomException as e:
            raise CustomException(e,sys)


