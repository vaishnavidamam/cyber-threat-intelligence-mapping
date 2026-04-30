#model.py
import torch
import torch.nn as nn
from transformers import RobertaModel
import config

class RobertaMultiLabel(nn.Module):
    def __init__(self, num_labels):
        super(RobertaMultiLabel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.MODEL_NAME)

        for name, param in self.roberta.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < 2:
                    param.requires_grad = False

        self.dropout = nn.Dropout(config.DROPOUT)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits