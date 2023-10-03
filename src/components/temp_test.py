import pandas as pd
import os
import sys

from src.pipeline.predict_pipeline import CustomData,PredictPipeline


train_data_path = os.path.join('artifacts',"train.csv")
temp_train_df = pd.read_csv(train_data_path, delimiter='~')
print(temp_train_df.shape)

# print(temp_train_df.head())

predict_pipeline=PredictPipeline()

results=predict_pipeline.predict("Amit Alcohol & Carbon Dioxide Ltd (Merged)")
print(results)