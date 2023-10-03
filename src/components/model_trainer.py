import os
import sys
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_df):
        try:
            logging.info("Initiated model training")

            model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
            passage_embedding = model.encode(train_df["stock_tags"].to_list())  

            similarity = cosine_similarity(passage_embedding)

            logging.info("Token encoding done using BERT!") 
            print(f"shape of similarity matrix: {similarity.shape}") 

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=similarity
            )
            
            return print(similarity.shape)
            



            
        except Exception as e:
            raise CustomException(e,sys)