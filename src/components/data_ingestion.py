import numpy as np
import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


## Data ingestion config:
@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('artifacts','raw.csv') #path for save whole dataset
    train_data_path=os.path.join('artifacts','train.csv') #path for save train dataset
    test_data_path=os.path.join('artifacts','test.csv') #path for save test dataset

## Dataingestion class: 
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def intiate_data_ingestion(self):
        try:
            df=pd.read_csv(os.path.join('notebooks/data','IRIS.csv'))
            logging.info('Datasets read as Pandas DataFrame')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            

            logging.info('Train test split')
            
            train_set,test_set=train_test_split(df,test_size=0.10,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info('Data ingestion is Completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Some Error occured into Data ingestion class')
            logging.info(e,sys)
        
        
        