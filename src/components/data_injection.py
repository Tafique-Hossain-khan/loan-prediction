from src.exception import CustomException
from src.logger import logging
import os
import pandas as pd 
import sys

from src.components.data_transformation import DataTransformation
class DataInjectionConfig:


    train_data_path:str = os.path.join('artifacts','train_data')
    test_data_path:str = os.path.join('artifacts','test_data')

class DataInjection:

    def __init__(self) -> None:

        self.injection_config = DataInjectionConfig()

    def initiate_data_ingection(self):
        try:

            logging.info('Staring Data injection')
            #code for reding the data from diffrent source
            train_data = pd.read_csv('notebook/data/train_df.csv')
            test_data = pd.read_csv('notebook/data/test_df.csv')

            logging.info("making the directory for storing the data")
            os.makedirs(os.path.dirname(self.injection_config.train_data_path),exist_ok=True)

            #convering the data into csv and storing into the required variable
            train_data.to_csv(self.injection_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.injection_config.test_data_path,index=False,header=True)
            
            logging.info('Data injection complited')

            return(
                self.injection_config.train_data_path,
                self.injection_config.test_data_path
            )

        except CustomException as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj = DataInjection()
    train_data,test_data=obj.initiate_data_ingection()
    #logging.info(train_data)
    obj2 = DataTransformation()
    obj2.initiate_data_transformation(train_data_path=train_data,test_data_path=test_data)





