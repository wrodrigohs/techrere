import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly_express as px
import json
import plotly
import json
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from my_model import LIABertClassifier
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.extras as extras
from psycopg2 import sql

app = Flask(__name__)

PATH = 'https://github.com/wrodrigohs/techrere/releases/download/model.pth/model.pth'
state_dict = torch.hub.load_state_dict_from_url(PATH, map_location=torch.device('cpu'))

model = BertModel.from_pretrained("bert-base-multilingual-cased")
model = LIABertClassifier(model, 3)
model.load_state_dict(state_dict)

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
    noticia = request.form['noticia']
    fig1, fig2 = plot(noticia)
    
    token=tokenizer(noticia, return_tensors='pt')
    out=model(input_ids=token['input_ids'])
    output = torch.argmax(out, dim=-1)[0].cpu().item()
    if output == 0:
      resultado = 'negativo'
    elif output == 1:
      resultado = 'positivo'
    else:
      resultado = 'neutro'

    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    # diff_date = datetime.now() - timedelta(days=7) 
    return render_template('index.html', 
                          graph1JSON=graph1JSON,
                          prediction_text='{}'.format(resultado),
                          noticia = noticia)

def get_noticia():
    conn = psycopg2.connect(host="localhost", 
      database="techrere", user="postgres", password="root")
    cur = conn.cursor()
    cur.execute("SELECT classe FROM noticias")
    df = cur.fetchall()
    return df
    # return rows

def plot(noticia):
    df = get_noticia()
    print('\n\n')
    print(df)
    print('\n\n')
    # if (noticia.lower() == 'itub3'):
    #   df_acao = pd.read_csv('https://raw.githubusercontent.com/wrodrigohs/techrere/main/itub3.csv')
    # elif (noticia.lower() == 'itub4'):
    #   df_acao = pd.read_csv('https://raw.githubusercontent.com/wrodrigohs/techrere/main/iutb4.csv')
    # elif (noticia.lower() == 'petr3'):
    #   df_acao = pd.read_csv('https://raw.githubusercontent.com/wrodrigohs/techrere/main/petr3.csv')
    # else:
    #   df_acao = pd.read_csv('https://raw.githubusercontent.com/wrodrigohs/techrere/main/petr4.csv')
  # Graph One
    df = px.data.medals_wide()
    fig1 = px.bar(df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input")
    fig2 = px.bar(df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input")
    return fig1, fig2

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0', port='5000')