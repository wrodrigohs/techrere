import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly_express as px
import json
import plotly
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from my_model import LIABertClassifier
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

    # graph1JSON = plot()
    # Graph One
    df = px.data.medals_wide()
    fig1 = px.bar(df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input")
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', 
                          #  graphJSON=graphJSON,
                           prediction_text='{}'.format(resultado))
# def plot():
#   import plotly.io as pio
#   pio.renderers.default = "iframe"
#   df = pd.DataFrame({
#       'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 
#       'Bananas'],
#       'Amount': [4, 1, 2, 2, 4, 5],
#       'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
#     })
#   fig = px.bar(df, x='Fruit', y='Amount', color='City', 
#     barmode='group')
#   fig.write_html("file.html", full_html=False, include_plotlyjs= False)
#   graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#   return graphJSON

if __name__ == "__main__":
    app.run(debug=True)