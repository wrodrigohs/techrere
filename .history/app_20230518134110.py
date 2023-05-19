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
    noticia = str(request.form.values())
    df = pd.read_csv('D:\\Downloads\\dfPetrobras.csv')
    
    token=tokenizer(df['text'], return_tensors='pt')
    out=model(input_ids=token['input_ids'])

    output = torch.argmax(out, dim=-1)[0].cpu().item()
    df['class'] = output
    df.to_csv('dfPetrobras.csv')
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    noticia = str(request.form.values())
    
    token=tokenizer(noticia, return_tensors='pt')
    out=model(input_ids=token['input_ids'])
    # print('\n\n{}\n\n'.format(out))
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

    # graph1JSON = plot()
    get_noticia()

    fig = plot()
    graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    diff_date = datetime.now() - timedelta(days=7) 
    return render_template('index.html', 
                          graph1JSON=graph1JSON,
                          prediction_text='{}'.format(resultado),
                          diff_date = diff_date)

def get_noticia():
    conn = psycopg2.connect(host="localhost", 
      database="techrere", user="postgres", password="root")
    cur = conn.cursor()
    cur.execute("SELECT * FROM empresa")
    rows = cur.fetchall()
    print(rows)
    # return rows

def plot():
  # Graph One
    df = px.data.medals_wide()
    fig1 = px.bar(df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input")
    return fig1

if __name__ == "__main__":
    app.run(debug=True)