import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


class BERTDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.data = {}
        for idx, row in df.iterrows():
            self.data[idx] = (row['review'], row['sentiment'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review, sentiment = self.data[idx]
        return review, torch.tensor(sentiment)


class BERT_IMDB(nn.Module):
    '''
    Fine-tuning DistillBert with two MLPs.
    '''

    def __init__(self, pretrained_type):
        super().__init__()

        num_labels = 2
        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_type, num_labels=num_labels)
        
        
        # TO-DO 2-1: Construct a classifier
        
        # BEGIN YOUR CODE
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.pretrained_model.config.hidden_size,2),
        #     nn.Linear(2,4),
        #     nn.Dropout(0.07),
        #     nn.Linear(4,4),
        #     nn.Dropout(0.05),
        #     nn.Linear(4,2),
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.pretrained_model.config.hidden_size,8),
            nn.Dropout(0.13),
            nn.Linear(8,4),
            nn.Linear(4,2),
        )
        
        
        # END YOUR CODE

    def forward(self, **pretrained_text):
        outputs = self.pretrained_model(**pretrained_text).last_hidden_state
        pretrained_output = outputs[:, 0, :]
        logits = self.classifier(pretrained_output)
        # print(logits.shape)
        return logits



class BERT():
    def __init__(self, pretrained_type, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_type)
        self.model = BERT_IMDB(pretrained_type).to(config['device'])
    
    def forward(self, text):    
        outputs = self.model(**text)
        return outputs

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()