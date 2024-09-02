import os 
import warnings
import sys 

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
# We need to import whatever library which we are using

import logging

# Creating the log
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_mertrics(actual,pred):
  rmse = np.sqrt(mean_squared_error(actual,pred))
  mae = mean_absolute_error(actual,pred)
  r2 = r2_score(actual,pred)
  return rmse, mae , r2

if __name__ =="__main__":
  warnings.filterwarnings('ignore')
  np.random.seed(40)
  
  # Reading the url
  csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
  # try:
  data = pd.read_csv(csv_url,sep=";")
  # except Exception as e:
  #   logger.exception(
  #     "Unable to download Traing CSV, "
  #   )
    
    # Train, test split
  train,test = train_test_split(data)
  train_x = train.drop(['quality'],axis=1)
  test_x = test.drop(['quality'],axis=1)
  train_y = train['quality']
  test_y = test['quality']
  
  alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
  l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
  # So here we are setting up default values
  with mlflow.start_run():
    lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio, random_state = 40)
    lr.fit(train_x,train_y)
    predicted_qualities = lr.predict(test_x)
    
    (rmse,mae,r2) = eval_mertrics(test_y,predicted_qualities)
    
    print("Elastic model(alpha = {:f},l1_ratio = {:f}):".format(alpha,l1_ratio))
    # :f specifies there should be floating number here
    print("RSME: %s"% rmse)
    print("MAE: %s"% mae)
    # %s means a string should be placed over here
    print("R2: %s"% r2)
    
    # Logging params
    mlflow.log_param("alpha",alpha)
    mlflow.log_param('l1_ratio',l1_ratio)
    # Logging meterics
    mlflow.log_metric('rmse',rmse)
    mlflow.log_metric('r2',r2)
    mlflow.log_metric('mae',mae)
    
    prediction = lr.predict(train_x)
    signature = infer_signature(train_x,prediction)
    
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
    if tracking_url_type_store != "file":
      mlflow.sklearn.log_model(
        lr,"model",registered_model_name="ElasticnetWineModel",signature=signature
      )
    else:
      mlflow.sklearn.log_model(lr,"model",signature=signature)
    
    
    