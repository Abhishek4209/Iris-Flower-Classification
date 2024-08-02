from src.logger import logging 
from src.exception import CustomException
from src.utils import load_objects
import os 
import sys
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            
            preprocessor=load_objects(preprocessor_path)
            model=load_objects(model_path)
            
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            
            return pred 
        
        
        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Some Error occured into predict function in prediction pipeline file')
            
class CustomData:
    def __init__(self,
                sepal_length:float,
                sepal_width:float,
                petal_length:float,
                petal_width:float):
        
        self.sepal_length=sepal_length
        self.sepal_width=sepal_width
        self.petal_length=petal_length
        self.petal_width=petal_width
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'sepal_length':[self.sepal_length],
                'sepal_width':[self.sepal_width],
                'petal_length':[self.petal_length],
                'petal_width':[self.petal_width]                
            }
        
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            # logging.info(f'{df}')
            return df     
        
        
        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Some error occured into Prediction pipeline')