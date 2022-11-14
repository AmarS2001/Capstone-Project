## Libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,LSTM, GRU, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import inspect
import pickle
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from IPython.display import HTML

import math
import pmdarima as pm

import seaborn as sns
sns.set_theme()
## Fastapi
app = FastAPI()

origins = [
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5000/dash',
    'http://127.0.0.1:5000/predict',
     

]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)







## Common Functions.
scaler = MinMaxScaler()

def connect(db):
    try:
        postgres_str = 'mysql+pymysql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(username="root",password="1234",ipaddress="localhost",port=3306,dbname=db)
        cnx = create_engine(postgres_str)
        return cnx
        
    except Exception as e:
        print(e)
        print("Make sure you changed the connections info in the connect() function")
        return None





## Data Class
class Data:
    def __init__(self,database,table_name):
        self.database = database
        self.table_name = table_name

    def connect(db):
        try:
            postgres_str = 'mysql+pymysql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(username="root",password="1234",ipaddress="localhost",port=3306,dbname=db)
            cnx = create_engine(postgres_str)
            return cnx
            
        except Exception as e:
            print(e)
            print("Make sure you changed the connections info in the connect() function")
            return None
            
    def get_table(self, table):
        conn=connect(self.database)
        if conn is None:
            print('Error connecting to MySQL database')
        
        meta_data = sqlalchemy.MetaData(bind=conn)
        sqlalchemy.MetaData.reflect(meta_data)
        table_sel = meta_data.tables[table]
        try:
            t = table_sel.select()
            result=conn.execute(t)
            cols=result.keys()
            vals=result.fetchall()
        
        except Exception as e:
            print(e)
            conn.rollback()
        vals=pd.DataFrame(vals)
        vals.columns = cols
        return vals

    def preprocess_data(self):
        def parse(x):
            return datetime.strptime(x, '%Y %m')

        def handle_categ(data,cols):
            return pd.get_dummies(data,columns=cols)
        
        dataset = self.get_table(self.table_name)
        dataset["date"] = dataset["Year"].astype(str) + " " + dataset["Month"].astype(str)
        dataset["date"] = dataset.apply(lambda x: parse(x["date"]), axis = 1)
        dataset = dataset.set_index("date")
        # dataset.drop(['ID','Year','Month','tempmax', 'tempmin','precipcover','windgust', 'windspeed', 'winddir','visibility', 'solarradiation', 'solarenergy', 'uvindex'], axis=1, inplace=True)
        #dataset = handle_categ(dataset,["Customer_Class"])
        dataset.drop(['Year','Month'], axis=1, inplace=True)
        temp = pd.DataFrame()
        temp["consumption"] = dataset["consumption"]
        dataset = dataset.drop(["consumption"],axis=1)
        dataset["consumption"] = temp["consumption"]
        dataset = dataset.astype(np.float32)
        return dataset




class FB_Prophet:
    model_name = "FB_Prophet"

    def __init__(self):
        self.train = 550
        self.model = None 
        

    def prepare_data(self, dataframe):
        dataframe = dataframe.reset_index()
        dataframe.rename(columns={'date':'ds', 'consumption':'y'}, inplace= True)
        return dataframe
        

    def model_train(self, dataframe):
        dataframe = self.prepare_data(dataframe)
        model = Prophet()
        # cols = dataframe.columns
        # fb_cols = [i for i in cols if i not in ['ds','y']]
        
        # for c in fb_cols:
        #     model.add_regressor(c, standardize= True)
        model.add_seasonality(name='monthly', period=30, fourier_order = 10)
        model.fit(dataframe)
        self.model = model

    def performance(self,y,y_hat):
        mse = mean_squared_error(y,y_hat)
        mae = mean_absolute_error(y,y_hat)
        mape = mean_absolute_percentage_error(y,y_hat)
        return mse, mae, mape

    def model_predict(self,dataframe):
        model = self.model
        data = self.prepare_data(dataframe)
        data = data.drop(['y'], axis=1 )
        ans = model.predict(data)
        return ans["yhat"]


class XGBoost:
    model_name = "XGBoost"

    def __init__(self):
        self.train = 550
        self.model = None 


    def prepare_data(self, dataframe):
        return dataframe

    def model_train(self, dataframe):
        dataframe = self.prepare_data(dataframe)
        model = xgb.XGBRegressor(n_estimators=1000,
                         learning_rate = 0.03,
                         max_depth = 15,
                         sub_sample = 0.9,
                         colsample_bytree=0.7,
                         missing=-999)
        y = dataframe["consumption"]
        X = dataframe.drop(["consumption"],axis=1)
        X_train, X_valid,y_train ,y_valid = train_test_split(X, y, test_size=0.01, random_state=42, shuffle=False)
        model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid,y_valid)], verbose=0)
        self.model = model


    def performance(self, y, y_hat):
        mse = mean_squared_error(y,y_hat)
        mae = mean_absolute_error(y,y_hat)
        mape = mean_absolute_percentage_error(y,y_hat)
        return mse, mae, mape

    def model_predict(self,dataframe):
        model = self.model
        data = self.prepare_data(dataframe)
        data = data.drop(["consumption"],axis=1)
        ans = model.predict(data)
        return ans


class ARIMA:
    model_name = "SARIMA"
    
    def __init__(self):
        self.model = None
    
    def prepare_data(self, dataframe):
        return dataframe

    def model_train(self,dataframe):
        dataframe = self.prepare_data(dataframe)
        self.model = pm.auto_arima(dataframe['consumption'], 
                      exogenous=dataframe[['population', 'temp', 'dew', 'humidity','precip', 
                                           'windspeed', 'cloudcover', 'sealevelpressure', 'gva']],
                      d= 1,
                      m = 12, 
                      seasonal=True,
                      alpha = 0.05,
                      test='adf',
                      error_action='ignore',  
                      suppress_warnings=True,
                      stepwise=True, 
                      trace=True)
    
    def performance(self):
       pass
        
    def model_predict(self,dataframe):

        n = dataframe.shape[0]
        forecast = self.model.predict(n_periods= n, exogenous= dataframe[['population', 'temp', 'dew', 
                                        'humidity','precip', 'windspeed', 'cloudcover', 'sealevelpressure', 'gva']], 
                                      return_conf_int=True)

        # forecast_df = pd.DataFrame(forecast[0], index = dataframe.index, columns=['consumption'])
        return forecast[0].values
        


# data = Data("data","jersey_data").preprocess_data()
# arima = ARIMA()
# train_data = data.iloc[:143,:]
# test_data = data.iloc[143:,:]
# arima.model_train(train_data)

# res = arima.model_predict(test_data)
# print(res)



class Ensemble:
    def __init__(self):
        self.model_name = "Ensemble"
        self.fb_model = None
        self.xg_model = None
        self.ar_model = None
        self.final = None

    def all_train_and_save(self,train_data):
        fbm = FB_Prophet()
        fbm.model_train(train_data)
        self.fb_model = fbm
        pickle.dump(fbm, open(fbm.model_name+".pkl","wb"))

        xgm = XGBoost()
        xgm.model_train(train_data)
        self.xg_model = xgm
        pickle.dump(xgm, open(xgm.model_name+".pkl","wb"))

        arm = ARIMA()
        arm.model_train(train_data)
        self.ar_model = arm
        pickle.dump(arm, open(arm.model_name+".pkl","wb"))


    def load_all_models(self):
        self.fb_model = pickle.load(open('FB_Prophet.pkl', 'rb'))
        self.xg_model = pickle.load(open('XGBoost.pkl','rb'))
        self.ar_model = pickle.load(open('SARIMA.pkl','rb'))

    def test_forecast(self,test_data):
        res1 = self.fb_model.model_predict(test_data).tolist()
        res2 = self.xg_model.model_predict(test_data).tolist()
        res3 = self.ar_model.model_predict(test_data)
        print(res3)
        
        final = pd.DataFrame()
        final["date"] = test_data.reset_index()["date"]
        #final["real_y"] = test_data["consumption"]
        final["fb_yhat"] = res1
        final["xg_yhat"] = res2
        final["ar_yhat"] = res3

        def mean_func(x):
            denom = 3
            sum = x["fb_yhat"] + x["xg_yhat"] + x["ar_yhat"]
            return sum/denom

        final["ensemble_yhat"] = final.apply(lambda x: mean_func(x),axis = 1)
        self.final=final
        return final

    def performance(self,test_data):
        perf = pd.DataFrame()
        self.final["real_y"] = test_data["consumption"].values.tolist()
        model_names = ["FB_prophet","XG_boost","Arima","ensemble"]
        model_mse  = []
        model_mse.append(mean_squared_error(self.final["real_y"], self.final["fb_yhat"]))
        model_mse.append(mean_squared_error(self.final["real_y"], self.final["xg_yhat"]))
        model_mse.append(mean_squared_error(self.final["real_y"], self.final["ar_yhat"]))
        model_mse.append(mean_squared_error(self.final["real_y"], self.final["ensemble_yhat"]))

        model_rmse  = []
        model_rmse.append((mean_squared_error(self.final["real_y"], self.final["fb_yhat"]))**0.5)
        model_rmse.append((mean_squared_error(self.final["real_y"], self.final["xg_yhat"]))**0.5)
        model_rmse.append((mean_squared_error(self.final["real_y"], self.final["ar_yhat"]))**0.5)
        model_rmse.append((mean_squared_error(self.final["real_y"], self.final["ensemble_yhat"]))**0.5)

        model_mae = []
        model_mae.append(mean_absolute_error(self.final["real_y"], self.final["fb_yhat"]))
        model_mae.append(mean_absolute_error(self.final["real_y"], self.final["xg_yhat"]))
        model_mae.append(mean_absolute_error(self.final["real_y"], self.final["ar_yhat"]))
        model_mae.append(mean_absolute_error(self.final["real_y"], self.final["ensemble_yhat"]))

        model_mape = []
        model_mape.append(mean_absolute_percentage_error(self.final["real_y"], self.final["fb_yhat"]))
        model_mape.append(mean_absolute_percentage_error(self.final["real_y"], self.final["xg_yhat"]))
        model_mape.append(mean_absolute_percentage_error(self.final["real_y"], self.final["ar_yhat"]))
        model_mape.append(mean_absolute_percentage_error(self.final["real_y"], self.final["ensemble_yhat"]))
        
        perf["Model_Name"] = model_names
        perf["mse"] = model_mse
        perf["rmse"] = model_rmse
        perf["mae"] = model_mae
        perf["mape"] = model_mape

        return perf


class Item(BaseModel):
    months: int

data = Data("data","jersey_data").preprocess_data()

ens = Ensemble()



@app.get("/train")
async def train():
    train_data= data.iloc[:120]
    ens.all_train_and_save(train_data)
    return {"message":"train"}

@app.get("/load")
async def load():
    ens.load_all_models()
    return {"message":"load"}

@app.get("/scores")
async def scores():
    test_data =  data.iloc[120:144]
    res = ens.test_forecast(test_data)
    ans = ens.performance(test_data)

    res.plot(y=['real_y','ensemble_yhat'])
    plt.show()
    
    print(res)
    print(ans)
    return {"htm_res": ans.to_html()}


@app.post("/predict")
async def predict(item : Item):
    predict_data = data.iloc[144:144 + item.months]
    res = ens.test_forecast(predict_data)
    print(res["ensemble_yhat"])
    predict_data["consumption"] = res["ensemble_yhat"].values.tolist()
    train_data= data.iloc[:144]

    plt.figure(figsize=(10,5))
    sns.lineplot(x=train_data.index, y=train_data['consumption'])
    sns.lineplot(x=predict_data.index, y=predict_data['consumption'])
    plt.legend(['history','prediction'],loc='upper left')
    plt.ylabel('Consumption -- Million Liters')
    plt.xlabel('Months')
    plt.figure(figsize=(20, 5))
    plt.savefig('..\Frontend\static\pics\plot.png')
 

    return {"htm_res": predict_data.to_html()}



