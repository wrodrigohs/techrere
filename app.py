import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from plotly.graph_objs import Layout
import json
import plotly
import plotly.io as pio
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

PATH = 'https://github.com/wrodrigohs/deploy-ds-model/releases/download/model/model.pth'
state_dict = torch.hub.load_state_dict_from_url(PATH, map_location=torch.device('cpu'))

model = BertModel.from_pretrained("bert-base-multilingual-cased")
model = LIABertClassifier(model, 3)
model.load_state_dict(state_dict)

# Colocando o modelo em modo de avaliação
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

@app.route('/api')
def api():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('api.html')

@app.route('/predictNews',methods=['POST'])
def predictNews():
    '''
    For rendering results on HTML GUI
    '''
    news = request.form['news']
    token=tokenizer(news, return_tensors='pt')
    out=model(input_ids=token['input_ids'])
    output = torch.argmax(out, dim=-1)[0].cpu().item()
    if output == 0:
      resultado = 'negativa'
    elif output == 1:
      resultado = 'positiva'
    else:
      resultado = 'neutra'
    
    return render_template('api.html', 
      prediction_text='A notícia foi classificada como {}'.format(resultado))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    stock = request.form.get('acao')

    fig1 = plot(stock)

    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # diff_date = datetime.now() - timedelta(days=7) 
    return render_template('index.html', 
                          graph1JSON=graph1JSON)

def get_news(stock):
    if (stock.lower() == 'itau'):
      query = 'SELECT data_noticia, classe FROM noticias WHERE empresa = 1 AND data_noticia > \'2023-01-01\''
    else:
      query = 'SELECT data_noticia, classe FROM noticias WHERE empresa = 0 AND data_noticia > \'2023-01-01\''

    conn = psycopg2.connect(host="localhost", 
      database="techrere", user="postgres", password="root")
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=['data_noticia', 'classe'])

    return df
    # return rows

def get_stock_info(stock):
    if (stock.lower() == 'itau'):
      query = 'SELECT nome, data_acao, fechamento FROM acao WHERE empresa = 1 AND data_acao > \'2023-01-01\''
    else:
      query = 'SELECT nome, data_acao, fechamento FROM acao WHERE empresa = 0 AND data_acao > \'2023-01-01\''

    conn = psycopg2.connect(host="localhost", 
      database="techrere", user="postgres", password="root")
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=['nome', 'data_acao', 'fechamento'])

    return df

def plot(stock):
    df_news = get_news(stock)
    df_stock = get_stock_info(stock)

    if (stock.lower() == 'itau'):
      df_stock1 = df_stock.drop(df_stock.loc[df_stock['nome'] == 'ITUB4'].index)
      df_stock2 = df_stock.drop(df_stock.loc[df_stock['nome'] == 'ITUB3'].index)
    else:
      df_stock1 = df_stock.drop(df_stock.loc[df_stock['nome'] == 'PETR4'].index)
      df_stock2 = df_stock.drop(df_stock.loc[df_stock['nome'] == 'PETR3'].index)

    df_grouped = df_news.groupby(['data_noticia', 'classe']).size().reset_index(name='count').sort_values(by=['data_noticia'], ascending=False)

    df_grouped = pd.pivot_table(df_grouped, values='count', index='data_noticia', columns='classe', fill_value=0)

    # Reordenar as colunas por ordem crescente
    df_grouped = df_grouped.reindex(sorted(df_grouped.columns), axis=1)

    # Resetar o índice
    df_grouped = df_grouped.reset_index()

    # Renomear as colunas
    df_grouped.columns = ['data_noticia', 'noticias_negativas', 'noticias_positivas', 'noticias_neutras']

    # Ordenar o DataFrame pela coluna 'data_noticia'
    df_grouped = df_grouped.sort_values('data_noticia', ascending=False)

    df_grouped.to_csv('novo.csv')

    my_layout = Layout(hoverlabel = dict(bgcolor = '#FFFFFF'), template = 'simple_white')

    # Criação do gráfico scatter
    fig = go.Figure(layout = my_layout)

    # Adicionar o traço de dispersão para notícias negativas
    fig.add_trace(go.Scatter(
        x=df_grouped['data_noticia'],
        y=df_grouped['noticias_negativas'],
        mode='lines+markers',
        name='Notícias negativas',
        marker=dict(color="#FF1610"),
        showlegend=True
    ))

    # Adicionar o traço de dispersão para notícias positivas
    fig.add_trace(go.Scatter(
        x=df_grouped['data_noticia'],
        y=df_grouped['noticias_positivas'],
        mode='lines+markers',
        name='Notícias positivas',
        marker=dict(color="#18ade0"),
        showlegend=True
    ))

    # Adicionar o traço de dispersão para notícias neutras
    fig.add_trace(go.Scatter(
        x=df_grouped['data_noticia'],
        y=df_grouped['noticias_neutras'],
        mode='lines+markers',
        marker=dict(color="#EBCBFF"),
        name='Notícias neutras',
        showlegend=True
    ))

    #Adicionar o gráfico de linha do fechamento das ações
    fig.add_trace(go.Scatter(
        x = df_stock1['data_acao'],
        y = df_stock1['fechamento'],
        hovertemplate='R$ %{y}',
        # hovertemplate = [f"{percent:.2f}%" for percent in facultativos['jovens']],
        marker=dict(color="#685c4a"),
        name='Preço da ação ITUB3' if df_stock1['nome'].iloc[0] == 'ITUB3' else 'Preço da ação PETR3',
        showlegend=True))
    
    fig.add_trace(go.Scatter(
        x = df_stock2['data_acao'],
        y = df_stock2['fechamento'],
        hovertemplate='R$ %{y}',
        # hovertemplate = [f"{percent:.2f}%" for percent in facultativos['jovens']],
        marker=dict(color="#51b797"),
        name='Preço da ação ITUB4' if df_stock2['nome'].iloc[0] == 'ITUB4' else 'Preço da ação PETR4',
        showlegend=True))

    fig.update_layout(hovermode="x unified", 
                      yaxis_range=[0,32], 
                      plot_bgcolor='rgba(255,255,255,255)',
                      xaxis=dict(showgrid=False, 
                                zeroline=False,), 
                      yaxis=dict(showgrid=False, 
                                zeroline=False,
                                tickvals=[0, 5, 10, 15, 20, 25, 30, 35],
                                # ticktext=['R$ 0', 'R$ 5', 'R$ 10', 'R$ 15',
                                # 'R$ 20', 'R$ 25', 'R$ 30', 'R$ 35'],
                                ),
                      xaxis_title="",
                      yaxis_title="",
                      legend_title='Legenda',
                      )
    
    fig.update_xaxes(rangeslider_visible=True,
                      rangeselector=dict(
                      buttons=list([
                          dict(count=1, label="1 mês", step="month", stepmode="backward"),
                          dict(count=2, label="2 meses", step="month", stepmode="backward"),
                          dict(count=3, label="3 meses", step="month", stepmode="backward"),
                          # dict(count=1, label="1y", step="year", stepmode="backward"),
                          dict(step="all", label="Todo o período")
                      ])
    ))

    return fig#, fig2

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='127.0.0.1', port='5000')
