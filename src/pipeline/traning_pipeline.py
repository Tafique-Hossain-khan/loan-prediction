from src.exception import CustomException
from src.logger import logging
import sys
from src.components.data_injection import DataInjection
from src.components.data_transformation import DataTransformation
from src.components.model_traner import ModelTraner

from src.pipeline.prediction_pipeline import CustomInput
from src.pipeline.prediction_pipeline import predictionPipeline


if __name__ == "__main__":
    try:
        obj = DataInjection()
        train_data,test_data=obj.initiate_data_ingection()
        #logging.info(train_data)
        obj2 = DataTransformation()
        train_arr,test_arr,file_path = obj2.initiate_data_transformation(train_data_path=train_data,test_data_path=test_data)

        obj3 = ModelTraner()
        obj3.train_model(train_arr,test_arr)

        obj4 = CustomInput('Male','Yes',	'1',	'Graduate',	'No',	4583,	1508.0,	128.0,	360.0,	1.0	,'Rural')
        featers = obj4.get_df()
        predict_obj = predictionPipeline()
        predict_obj.predict(featers)
        #logging.info(predict_obj.predict(featers))

    except CustomException as e:
        raise CustomException(e,sys)
