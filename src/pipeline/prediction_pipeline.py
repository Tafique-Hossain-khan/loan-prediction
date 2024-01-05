import os
import sys
from src.logger import logging 
from src.exception import CustomException
from src.utils import load_object
import pandas as pd


class predictionPipeline:
    
    def __init__(self) -> None:
        pass

    def predict(self,feautres):

        try:
            logging.info("creating a prediction pipeline")
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            encoded_feature = preprocessor.transform(feautres)
            prediction = model.predict(encoded_feature)

            logging.info("Prediction pipeline Done!")
            return prediction

        except CustomException as e:
            raise CustomException(e,sys)
        

class CustomInput:

    def __init__(self,Gender, Married, Dependents, Education,
       Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
       Loan_Amount_Term, Credit_History, Property_Area) -> None:
        
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome  = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area

    
    def get_df(self):

        logging.info("creating a custom dataset")
        try:
            custom_input = {
                'Gender':[self.Gender],
                'Married':[self.Married],
                'Dependents':[self.Dependents],
                'Education':[self.Education],
                'Self_Employed':[self.Self_Employed],
                'ApplicantIncome':[self.ApplicantIncome],
                'CoapplicantIncome':[self.CoapplicantIncome],
                'LoanAmount':[self.LoanAmount],
                'Loan_Amount_Term':[self.Loan_Amount_Term],
                'Credit_History':[self.Credit_History],
                'Property_Area':[self.Property_Area],     
            }
            logging.info(pd.DataFrame(custom_input))
            
            return pd.DataFrame(custom_input)

        except CustomException as e:
            raise CustomException(e,sys)
        
