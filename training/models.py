import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        for param in self.bert.parameters():
            param.requires_grad = False #Makes the params not trainable. 
        
        self.projection = nn.Linear(768, 128)
        
    def forward(self, input_ids, attention_mask):
        #Extract Bert Embeddings
        outputs = self.bert(input_ids, attention_mask) 
        #Turns the token ids into embeddings 
        
        #use [CLS] token 
        pooled_output = outputs.pooler_output
        
        return self.projection(pooled_output)
        
        
class VideoEncoder(nn.Module):
    def __init__(self):
        self.backbone = vision_models