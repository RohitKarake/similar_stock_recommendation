import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

ps = PorterStemmer()

def remove_stopwords(obj):
        l1 = []
        tokens = word_tokenize(obj)
        for token in tokens:
            if token not in stopwords.words("english"):
                l1.append(token)
        return " ".join(l1)

def stemming(text):
    result = []
    for i in text.split():
        result.append(ps.stem(i))
    return " ".join(result)

@dataclass
class DataTransformationConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    target_data_path: str=os.path.join('artifacts',"target.csv")

# Custom transformer for text cleaning
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, df, str_to_replace="\n"):
        return self
    
    def transform(self, df, str_to_replace="\n"):
        # df["coalesce"] = df["coalesce"].str.lower()
        df["coalesce"] = df["coalesce"].apply(lambda x: x.replace("\r\n\r\n", ""))
        df["coalesce"] = df["coalesce"].apply(lambda x: x.replace("\r\n", ""))
        df["coalesce"] = df["coalesce"].apply(lambda x: x.split())

        df["sect_name"] = df["sect_name"].apply(lambda x: x.replace(" ", ""))
        df["sect_name"] = df["sect_name"].apply(lambda x: x.split())
    
        df["stock_tags"] = df["sect_name"] + df["coalesce"]
        df["stock_tags"] = df["stock_tags"].apply(lambda x: " ".join(x))
        df["stock_tags"] = df["stock_tags"].str.lower()
        df["lname"] = df["lname"].str.lower()
        
        return df

# Custom transformer for tokenization and further processing
class Tokenizer(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        df["stock_tags"] = df["stock_tags"].apply(remove_stopwords)
        df["stock_tags"] = df["stock_tags"].apply(stemming)

        # lemmatizer = WordNetLemmatizer()
        # df["stock_tags"] = df["stock_tags"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        return df

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:

            preprocessor = Pipeline([
                ('cleaner', TextCleaner()),
                ('tokenizer', Tokenizer())
                ])

            logging.info(f"Pre-processing pipeline passed!")


            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,raw_path):

        try:
            raw_df=pd.read_csv(raw_path, delimiter='~')

            logging.info("Read raw data completed!")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            columns_to_drop = ["isin", "co_code", "co_code.1"]
            input_feature_train_df=raw_df.drop(columns=columns_to_drop, axis=1)


            logging.info(
                f"Applying preprocessing object on raw dataframe."
            )

            input_feature_train_df=preprocessing_obj.fit_transform(input_feature_train_df)


            logging.info(f"Saved preprocessing object.")

            input_feature_train_df.to_csv(self.data_transformation_config.train_data_path, sep = "~", index=True,header=True)

            input_feature_train_df["lname"].to_csv(self.data_transformation_config.target_data_path,index=True,header=True)

            return (
                input_feature_train_df,
                self.data_transformation_config.train_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
