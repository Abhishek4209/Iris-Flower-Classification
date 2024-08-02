import numpy as np
import pandas as pd
import os 
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.pipelines.prediction_pipeline import CustomData,PredictionPipeline
from flask import Flask,render_template,request


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    else:
        data=CustomData(
            sepal_length=float(request.form.get('sepal_length')),
            sepal_width=float(request.form.get('sepal_width')),
            petal_length=float(request.form.get('petal_length')),
            petal_width=float(request.form.get('petal_width'))
                        )
        
        
        logging.info(data)
        final_new_data=data.get_data_as_dataframe()
        logging.info(final_new_data)
        predict_pipeline=PredictionPipeline()
        pred=predict_pipeline.predict(final_new_data)
        results=(pred[0])
        return render_template('result.html',final_result=results)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')