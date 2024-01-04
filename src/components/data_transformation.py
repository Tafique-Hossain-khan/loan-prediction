from src.exception import CustomException
from src.logger import logging
import os
import sys
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object


class DataTransformationConfig:

    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:

   

    
        def __init__(self) -> None:
            self.data_transformation_config = DataTransformationConfig()

        def get_data_transformed_obj(self,train_data,test_data):

            try:
                logging.info("i am started ")
                #logging.info(train_data_path)
                #train_data = pd.read_csv(train_data_path)
                #test_data = pd.read_csv(test_data_path)
                
                logging.info(train_data.head(1))

                #train_data.drop(columns=['Loan_ID'],axis=1,inplace=True)
                #test_data.drop(columns=['Loan_ID'],axis=1,inplace=True)
                

                logging.info('Data Transformation initiated')

                #get the cat and num col

                cat_col = [ 'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area', 'Loan_Status']
                
                num_col = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History']
                
                logging.info('Handeling the null values of all the columns')
                
                null_num_col = ['Credit_History', 'LoanAmount', 'Loan_Amount_Term']
                null_cat_col = ['Self_Employed', 'Dependents', 'Gender', 'Married']

                def fill_nan_cat(df,col):
                    for i in col:
                        df[i].fillna(df[i].mode()[0],inplace= True)
                def fill_nan_num(df,col):
                    for i in col:
                        df[i].fillna(df[i].median(),inplace=True)
                #fill nan for cat col
                fill_nan_cat(train_data,null_cat_col)
                fill_nan_cat(test_data,null_cat_col)
                # fill nan for num col
                fill_nan_num(train_data,null_num_col)
                fill_nan_num(test_data,null_num_col)

                logging.info('Missing value Done!')

                logging.info("Handeling outliers")
                def handel_outliers(df,col):

                    for i in col:
                        upper_lim = df[i].mean() + 3*df[i].std()
                        lowe_lim = df[i].mean() - 3*df[i].std()

                        df[i] = np.where(
                            df[i] > upper_lim,
                            upper_lim,
                            np.where(
                                df[i] < lowe_lim,
                                lowe_lim,
                                df[i]
                            )
                        )
                handel_outliers(train_data,num_col)
                handel_outliers(test_data,num_col)
                
                logging.info(train_data.isnull().sum())
                logging.info('Outilers Handeled sucessfully!')

                
                logging.info("upsampeling the data")
                #performing upsampeling

                #performing upsampeling
                #performing upsampeling

                from imblearn.over_sampling import RandomOverSampler
                X = train_data.drop(columns=['Loan_Status'],axis='columns')
                y = train_data['Loan_Status']
                os = RandomOverSampler(random_state=42,sampling_strategy=0.8)
                X_res,y_res = os.fit_resample(X,y)
                new_df = pd.concat([X_res,y_res],axis=1) #new upsampled dataframe

                logging.info("Handeling the skewness")

                right_skew = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
                left_skew = ['Loan_Amount_Term',]
                transformer1 = FunctionTransformer(np.log1p)

                def square_transform(x):
                    return np.square(x)

                transformer2 = FunctionTransformer(square_transform)

                for col in right_skew:
                    new_df[col] = transformer1.transform(new_df[col])
                for col in left_skew:
                    new_df[col] = transformer2.transform(new_df[col])

                #logging.info(train_data.skew())
                logging.info('skewness Done!')

                logging.info("Creating pipeline")
                nominal_pipeline = Pipeline([
                    ('ohe', OneHotEncoder(drop='first', sparse=False, dtype=np.int32))
                ])
                ordinal_pipeline = Pipeline([
                    ('ord', OrdinalEncoder(categories=[['Graduate', 'Not Graduate']]))  
                ])
                numerical_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                ])
                new_num_col = new_df.select_dtypes(include=['float','int']).columns

                trnf = ColumnTransformer([
                    ('nominal',nominal_pipeline,['Gender', 'Married', 'Dependents', 'Self_Employed',
                    'Property_Area']),
                    ('ordinal',ordinal_pipeline,['Education']),
                    
                    ('num',numerical_pipeline,new_num_col)
                ])

                logging.info("Data Transformation Done!")

                return trnf

            except CustomException as e :
                    raise CustomException(e,sys)   
        

        def initiate_data_transformation(self,train_data_path,test_data_path):
             
            try:
                train_data = pd.read_csv(train_data_path)
                test_data = pd.read_csv(test_data_path)

                train_data.drop(columns=['Loan_ID'],axis=1,inplace=True)
                test_data.drop(columns=['Loan_ID'],axis=1,inplace=True)
                logging.info(train_data.head(1))

                logging.info('Getting the transformation object')

                preprocessor_obj = self.get_data_transformed_obj(train_data,test_data)

                target_col = ['Loan_Status']

                logging.info('Spliting the data')
                logging.info(train_data.columns)

                input_feature_train_df = train_data.drop(target_col,axis=1)
                target_feature_train_df = train_data[target_col]

                input_feature_test_df = test_data

                logging.info('Transformation start')
                input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


                train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr =  input_feature_test_arr

                logging.info("Data Transformation Done!")

                save_object(self.data_transformation_config.preprocessor_obj_file_path,
                          preprocessor_obj )

                return(
                     train_arr,test_arr,
                     self.data_transformation_config.preprocessor_obj_file_path
                 )
            


            except CustomException as e:
                raise CustomException(e,sys)
             


    




