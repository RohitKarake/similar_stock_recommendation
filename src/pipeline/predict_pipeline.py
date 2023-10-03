import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, stock):
        try:
            target_data_path = os.path.join('artifacts',"target.csv")
            stock_df = pd.read_csv(target_data_path)
            model_path=os.path.join("artifacts","model.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            # print(model)
            print("After Loading")
            similar_stocks = self.recom_with_bert(stock, model, stock_df)
            return similar_stocks
        
        except Exception as e:
            raise CustomException(e,sys)            
        
        
    def recom_with_bert(self, stock, model, stock_df):
        stock = stock.lower()
        stock_list = stock_df["lname"]
        if stock in stock_list.to_list():
            index = stock_df[stock_df["lname"] == stock].index[0]
        else:
            index = self.find_closest(stock, stock_df)
            
        distances = sorted(list(enumerate(model[index])), reverse= True, key= lambda x: x[1])
        for i in distances[1:10]:
            print(stock_df.iloc[i[0]].lname)

        
    def find_closest(self, stock, stock_df):
        spliter = stock_df["lname"].str.split(" ")
        i = 0
        for val in spliter:
            if stock in val:
                print(stock_df["lname"][i])
                return i
            i += 1



class CustomData:
    def __init__(  self,
        stock_name: str):

        self.stock_name = stock_name

    def get_data_as_json(self):
        try:
            custom_data_input_dict = {
                "stock_name": self.stock_name
            }

            return custom_data_input_dict

        except Exception as e:
            raise CustomException(e, sys)