import argparse
import warnings
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from bert import BERT, BERTDataset
from rnn import RNN, RNNDataset
from ngram import Ngram
from preprocess import preprocessing_function
import gc

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device, torch.__version__)
def prepare_data():
    # do not modify
    df_train = pd.read_csv('./data/IMDB_train.csv')
    df_test = pd.read_csv('./data/IMDB_test.csv')
    #df_train = df_train[:500]
    #df_test = df_test[:100]
    return df_train, df_test

def get_argument():
    # do not modify
    opt = argparse.ArgumentParser()
    opt.add_argument("--model_type",
                        type=str,
                        choices=['ngram', 'RNN', 'BERT'],
                        required=True,
                        help="model type")
    opt.add_argument("--preprocess",
                        type=int,
                        help="whether preprocessing, 0 means no and 1 means yes")
    opt.add_argument("--part",
                        type=int,
                        help="specify the part")

    config = vars(opt.parse_args())
    return config

def first_part(model_type, df_train, df_test, N):
    # load and train model
    if model_type == 'ngram':
        model = Ngram(N)
        model.train(df_train)
    else:
        raise NotImplementedError

    # test performance of model
    perplexity = model.compute_perplexity(df_test)
    print("Perplexity of {}: {}".format(model_type, perplexity))

    return model

def second_part(model_type, df_train, df_test, N):


    # training configure
    rnn_config = {
        'batch_size': 8,
        'epochs': 15,
        'lr': 5e-4,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }

    bert_config = {
        'batch_size': 6,
        'epochs': 1,
        'lr': 2e-5,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
   
    }
    

    
    # load and dataset and set model
    if model_type == 'ngram':
        model = first_part(model_type, df_train, df_test, N)
        pass

    elif model_type == 'RNN':
        # pad and truncate for this batch
        def collate_fn(data):
            sequences, labels = zip(*data)
            sequences, labels = list(sequences), list(labels)
            sequences = pad_sequence(sequences, batch_first=False, padding_value=0)
            sequences = sequences[:512, :]
            return sequences, torch.tensor(labels)
        
        train_data = RNNDataset(df_train)
        test_data = RNNDataset(df_test, train_data)
        train_dataloader = DataLoader(train_data, collate_fn=collate_fn, batch_size=rnn_config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_data, collate_fn=collate_fn, batch_size=1, shuffle=False)
        rnn_config['vocab_size'] = len(train_data.vocab_word2int)
        model = RNN(config = rnn_config)

    elif model_type == 'BERT':
        model = BERT('distilbert-base-uncased', config = bert_config)

        def collate_fn(data):
            sequences, labels = zip(*data)
            sequences, labels = list(sequences), list(labels)
            sequences = model.tokenizer(sequences, padding=True, truncation=True,max_length=512, return_tensors="pt")
            return sequences,torch.tensor(labels)

        train_data = BERTDataset(df_train)
        test_data = BERTDataset(df_test)
        train_dataloader = DataLoader(train_data, batch_size= bert_config['batch_size'], collate_fn = collate_fn)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn = collate_fn)
               
    else:
        raise NotImplementedError

    # train model
    if model_type == 'ngram':
        model.train_sentiment(df_train, df_test)
    elif model_type == 'BERT' or model_type == 'RNN':    
        train(
            model_type=model_type,
            model=model,
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader,
            optimizer=torch.optim.Adam(model.parameters(), lr=model.config['lr']),
            loss_fn=nn.CrossEntropyLoss().to(model.config['device']),
            config=model.config
        )
    else:
        raise NotImplementedError

def train(model_type, model, train_dataloader, test_dataloader, optimizer, loss_fn, config):
    '''
    total_loss: the accumulated loss of training
    labels: the correct labels set for the test set
    pred: the predicted labels set for the test set
    '''
    # TO-DO 2-2: Implement the training function
    # BEGIN YOUR CODE
    epochs = config['epochs']
    print('strat training! mdfk')
    for epoch in range(epochs): # This is pseudo code
        model.train() 
        # training stage
        print('total batches:',len(train_dataloader))
        trainLoss = 0
        for i, (x, y) in enumerate((train_dataloader)): # This is pseudo code
            x = x.to(device) 

            y = [[1 if i == val else 0 for i in range(2)] for val in y]
            y = torch.tensor(y).float()
            y = y.to(device)

            if i % 1000 == 0: 
                
                print('    batch',i,'| loss:',trainLoss/1000)
                trainLoss = 0
            optimizer.zero_grad()
            
            logit = model.forward(x)
            # logit = torch.tensor(logit).float() 
            # print(y, logit)
            # pred = torch.roung(torch.sigmoid(logit))
            # Pred = logit 
            
            loss = loss_fn(logit, y)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()
            torch.cuda.empty_cache()
            gc.collect()
            # break
            

        model.eval()
        # testing stage
        total_loss = 0
        labels = [y for x,y in test_dataloader]
        preds = []
        with torch.inference_mode():
            for i, (x, y) in enumerate(test_dataloader): # This is pseudo code
                if i % 10000 == 0: 

                    print('testing...')
                x = x.to(device) 

                y = [[1 if i == val else 0 for i in range(2)] for val in y] 
                y = torch.tensor(y).float()

                y = y.to(device) 
                logit = model.forward(x)
      

                if model_type == 'RNN':
                    logit = logit.unsqueeze(dim=0)

                pred = torch.softmax(logit, dim = 1).argmax(dim = 1) 
                # num_ones = (pred == 1).sum().item()
                # num_zeros = (pred == 0).sum().item()
                # print(num_ones, num_zeros)
                # Check if the number of ones is greater than the number of zeros
                # pred = 1 if num_ones > num_zeros else 0
                preds.append(pred)
                # print(logit,y)
                loss = loss_fn(logit, y)
                total_loss += loss.item()
        labels = torch.tensor(labels).cpu().numpy()
        
        preds = torch.cat(preds, dim=0).cpu().numpy() if model_type == 'BERT' else torch.tensor(preds).cpu().numpy()
        
        precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average='macro', zero_division=1)

        avg_loss = total_loss/len(train_dataloader)
        print(f"Epoch: {epoch}, F1 score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Loss: {avg_loss:.4f}")
        print('.')
    # END YOUR CODE
    
if __name__ == '__main__':
    
    # get argument
    config = get_argument()
    model_type, preprocessed = config['model_type'], config['preprocess']
    N = 2 # we only use bi-gram in this assignment, but you can try different N

    # read and prepare data
    df_train, df_test = prepare_data()
    label_mapping = {'negative': 0, 'positive': 1}
    df_train['sentiment'] = df_train['sentiment'].map(label_mapping)
    df_test['sentiment'] = df_test['sentiment'].map(label_mapping)

    # feel free to add more text preprocessing method
    if preprocessed:
        df_train['review'] = df_train['review'].apply(preprocessing_function)
        df_test['review'] = df_test['review'].apply(preprocessing_function)

    if config['part'] == 1:
        # part 1: implement bi-gram model
        first_part(model_type, df_train, df_test, N)
    elif config['part'] == 2:
        # part 2: implement your nn model from scratch
        # part 3: implement your nn model from other pre-trained model
        second_part(model_type, df_train, df_test, N)