from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union

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