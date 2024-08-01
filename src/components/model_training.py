import numpy as np
import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass
from src.utils import save_objects
from src.utils import evaluate_model


## Model trainig config class:
@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    

## Model Training class:
class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Spliting Dependent and Independent variables from train and test data')
            X_train,y_train,X_test,y_test=(
                
                
            )
            models={
    'LogisticRegression':LogisticRegression(),
    'SVC':SVC(),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'RandomForestClassifier':RandomForestClassifier(criterion='log_loss',max_depth=5,n_estimators=10),
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'AdaBoostClassifier':AdaBoostClassifier(),
    'KNeighborsClassifier':KNeighborsClassifier(algorithm='auto',p=2,n_neighbors=7)
                    }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_objects(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
            logging.info('Best Model :{best_model} with accuracy : {best_model_score}')
        
        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Some Error occured into initate model training class')
            
