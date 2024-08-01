import numpy as np
import pandas as pd
import os 
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from dataclass import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


## Data Transformation config class:
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
## Data Transformation class:
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            
            numeric_features=[]
            categorical_features=[]
            
            logging.info('Pipeline Intitiated')
            
            num_pipeline=Pipeline(steps=[
            ('Imputer',SimpleImputer(strategy='median')),
            ('Scaler',StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
            ('Imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(sparse_output=False)),
            ('Scaler',StandardScaler())
    
            ])
            
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_columns.columns.tolist()),
            ('cat_pipeline',cat_pipeline,categorical_columns.columns.tolist())
            ])
            
            logging.info('Pipeline Completed')

            return preprocessor
        
            
        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Some Error Occured into data transformation class')
            
            
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train set and test data completed')
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj=self.get_data_transformation_object()
            
            target_column_name='species'
            drop_columns=[target_column_name]
            
            # Feature devide  into independet and depedent features
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name] 

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name] 
            
            ## apply the transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datsets.")
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor pickle in create and saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )           
        
        except Exception as e:
            raise CustomException(e,sys) 
            logging.info('Some Error occured into initiate_data_transformation')

