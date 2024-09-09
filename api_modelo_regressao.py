import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class request_body(BaseModel):
    horas_estudo : float

# Carregar modelo para predição
modelo_pontuacao = joblib.load('./modelo_regressao.pkl')

@app.post('/predict')
def predict (data : request_body):
  # Preparar os dados para predição
  input_feature = [[data.horas_estudo]]
  
  # Fazer a predição
  y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)
  
  return {"pontuacao_teste" : y_pred.tolist()}