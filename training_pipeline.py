import numpy as np
import pandas as pd
import os 
import sys
from src.logger import logging
from src.exceptions import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining




if __name__ == '__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.intiate_data_ingestion()
    print(train_data_path,test_data_path)
    logging.info("Training is Completed")
    
    # data_transformation=DataTransformation()
    # train_arr,test_arr=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    # logging.info('Data Transformation is Completed')
    
    
    # model_trainer=ModelTraining()
    # model_trainer.initate_model_training(train_arr,test_arr)
    # logging.info('Training is Completed')
    
    