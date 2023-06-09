# -*- coding: utf-8 -*-
"""createModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DYzWgW9KnQ-SOTBZ7n-2LnC7zL05NP_a
"""

#pip install transformers -q

import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import pandas as pd
#from google.colab import drive

import numpy as np

from transformers import BertTokenizer, BertModel, BertPreTrainedModel, get_scheduler

from typing import List, Optional, Tuple, Union

import sklearn.model_selection as model_selection
from sklearn import metrics

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")

class LIABertClassifier(nn.Module):
    def __init__(self,model,num_labels):
        super(LIABertClassifier,self).__init__()
        self.bert = model
        self.config = model.config
        self.num_labels = num_labels
        self.cls = nn.Linear(self.config.hidden_size,200)
        self.dropout = nn.Dropout(p=0.5)
        self.cls2 = nn.Linear(200,num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        ) ->Tuple[torch.Tensor]:

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0][:,0,:]
        prediction = self.cls(sequence_output)
        prediction = self.dropout(prediction)
        prediction = self.cls2(prediction)
        return prediction

model = LIABertClassifier(model,3)

#drive.mount('/content/drive')

df_train = pd.read_csv("/content/dataset.csv")

df_train.head(10)

y_train[y_train == 'negative'] = 0
y_train[y_train == 'positive'] = 1
y_train[y_train == 'neutral'] = 2

y_train = np.array(y_train.tolist(), dtype=int)

xtrain, xval, ytrain, yval = model_selection.train_test_split(x_train, y_train, test_size=0.30, random_state=42,shuffle=True)

train_encodings = tokenizer(xtrain.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
val_encodings = tokenizer(xval.tolist(), truncation=True, padding=True,max_length=512, return_tensors='pt')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        label = torch.tensor(self.labels[idx])
        return (item,label)

    def __len__(self):
        return len(self.labels)

ret=train_encodings.items()

ds_train = MyDataset(train_encodings,ytrain)
ds_val   = MyDataset(val_encodings,yval)

dl_train = DataLoader(ds_train,batch_size=8)
dl_eval  = DataLoader(ds_val,batch_size=8)

x,y = next(iter(dl_train))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

device

batch = {k: v.to(device) for k, v in x.items()}

model.to(device)

out = model(**batch)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10

loss_fct = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    lepochs = []
    for batch,y in dl_train:
        batch = {k: v.to(device) for k, v in batch.items()}
        y     = y.to(device)
        outputs = model(**batch)
        #print(outputs,y)
        loss = loss_fct(outputs,y)
        loss.backward()
        lepochs.append(loss.cpu().item())
        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
    print(np.mean(lepochs))

model.eval()

ytrue = []
ypred = []
for batch,y in dl_eval:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = torch.argmax(outputs, dim=-1)
    ytrue += y.tolist()
    ypred += predictions.cpu().tolist()

torch.save(model.state_dict(),'/content/model.pth')

#backup = torch.load('/content/model.pth')

#model.load_state_dict('/content/model.pth')

x,y = next(iter(dl_eval))

batch = {k: v.to(device) for k, v in x.items()}

with torch.no_grad():
    outputs = model(**batch)

metrics.confusion_matrix(ytrue,ypred)

print(metrics.classification_report(ytrue,ypred))