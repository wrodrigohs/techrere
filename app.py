import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly_express as px
import plotly
import json
import pickle
import os
import pathlib
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from my_model import LIABertClassifier
app = Flask(__name__)

model = BertModel.from_pretrained("bert-base-multilingual-cased")
model = LIABertClassifier(model, 3)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
# Carregando os parâmetros do modelo
# parameters = torch.load('model.pth', map_location=torch.device('cpu'))

# Criando uma instância do modelo
# model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")

# Atribuindo os parâmetros carregados à instância do modelo, ignorando as chaves incompatíveis
# state_dict = {k: v for k, v in parameters.items() if k in model.state_dict()}
# model.load_state_dict(state_dict, strict=False)

# Atribuindo os parâmetros carregados à instância do modelo
# model.load_state_dict(parameters)

# Colocando o modelo em modo de avaliação
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    noticia = str(request.form.values())
    
    token=tokenizer(noticia, return_tensors='pt')
    out=model(input_ids=token['input_ids'])
    print('\n\n{}\n\n'.format(out))
    # logits = out.logits
    # output = torch.argmax(logits, dim=-1).item()

    output = torch.argmax(out, dim=-1)[0].cpu().item()
    if output == 0:
      resultado = 'negativo'
    elif output == 1:
      resultado = 'positivo'
    else:
      resultado = 'neutro'
    # final_features = [np.array(int_features)]
    # prediction = model.predict(int_features)

    # output = prediction[0]

    graphJSON = plot()

    return render_template('index.html', 
                          #  graphJSON=graphJSON,
                           prediction_text='{}'.format(resultado))
def plot():
  import plotly.io as pio
  pio.renderers.default = "iframe"
  df = pd.DataFrame({
      'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 
      'Bananas'],
      'Amount': [4, 1, 2, 2, 4, 5],
      'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
    })
  fig = px.bar(df, x='Fruit', y='Amount', color='City', 
    barmode='group')
  fig.write_html("file.html", full_html=False, include_plotlyjs= False)
  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON

if __name__ == "__main__":
    app.run(debug=True)