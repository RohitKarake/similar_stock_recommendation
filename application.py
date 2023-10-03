from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/similar_stocks',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('stock_name')

        )
        input_json=data.get_data_as_json()
        print(input_json)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(input_json["stock_name"])
        print(f"after Prediction results: {results}")
        return render_template('home.html',results=results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)        