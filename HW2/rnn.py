import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter

class RNNDataset(Dataset):
    def __init__(self, df, train_data=None):
        super().__init__()

        self.reviews = df['review'].values
        self.sentiments = df['sentiment'].values

        # tokenize each sentence
        tokenzier = get_tokenizer("basic_english")
        for idx in range(len(self.reviews)):
            self.reviews[idx] = ' '.join(tokenzier(self.reviews[idx]))

        if train_data is None:
            # build vocab
            words = ' '.join(self.reviews)
            words = words.split()
            counter = Counter(words)
            vocab = sorted(counter, key=counter.get, reverse=True)

            self.vocab_int2word = dict(enumerate(vocab, 2))
            self.vocab_int2word[0] = '<PAD>'
            self.vocab_int2word[1] = '<UNK>'
            self.vocab_word2int = {word: id for id, word in self.vocab_int2word.items()}
        else:
            self.vocab_int2word = train_data.vocab_int2word
            self.vocab_word2int = train_data.vocab_word2int

        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review, sentiment = self.reviews[idx], self.sentiments[idx]
        review = [self.vocab_word2int.get(word, 1) for word in review.split()]
        return (torch.tensor(review), torch.tensor(sentiment))

class YourModel(nn.Module):
    def __init__(
            self,
            inputSize,
            hiddenSize,
            numLayer,
            embededDim
        ):
        super().__init__()

        
        '''
        When we want to build a model using PyTorch, we first create a class and make it inherit from torch.nn.Module. 
        Then, there are two important methods that we need to define ourselves: init() and forward(). 
        In the init() function, we need to define which components our model consists of. 
        In forward(), we need to define how the input data to the model interacts with the components
        defined in init() to generate the final output.
        
        For example, in a sentiment classification model, we typically need at least:
        
        1. torch.nn.Embedding()
        2. torch.nn.RNN()
        3. torch.nn.Linear()
        
        Note that in this part: 

        1. Please use torch.nn to create the components required for your model. 
        
        2. please do not use any pre-trained models or components. In other words, the parameters of 
        the components you define must be randomly initialized rather than pre-trained by others.
        '''
        # TO-DO 3-2: Determine which modules your model should consist of
        # BEGIN YOUR CODE

        # import math
       
        # self.model = nn.Embedding(inputSize, s)
            
        #         # END YOUR CODE
        # self.rnn = nn.RNN(input_size=s, hidden_size = hiddenSize, num_layers=numLayer) 
        # self.lin = nn.Linear(hiddenSize,2)
        self.embedding = nn.Embedding(inputSize, embededDim, padding_idx=0)
        self.LSTM = nn.LSTM(embededDim, hiddenSize, numLayer, dropout=0.4)
        self.lin = nn.Linear(in_features=hiddenSize, out_features=2)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None
      
        
    def forward(self, text):

        
        '''
        In the forward() function, we decide which operations the input will undergo to generate the output.
        For example, in a sentiment classification model, the input usually goes through
        Embedding() -> RNN() -> Linear() in sequence to obtain the final output.
        '''
        # TO-DO 3-3: Determine how the model generates output based on input
        # text = torch.transpose(text, 0, 1).long()
        # output = self.model(text)

        # output, _ = self.rnn(output)
        # output = self.lin(output)
        # output = torch.squeeze(output[:,-1,:],dim = 1).float()
        # # print(output.shape)

        # return output


        # lengths = x[0,:]
        # reviews = x[1:,:]
        embeds = self.embedding(text)
        output, _ = self.LSTM(embeds)
        output = output[-1,:,:]
        output = self.lin(output)
        # out = out[lengths - 1, range(len(lengths))]
        return self.sig(output.squeeze())
        # END YOUR CODE
    
class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        
        '''
        Simply pass any parameters you want to initialize your model.
        Of course, if you want to hardcode all the parameters in the definition of the 
        YourModel class, then you don't need to pass any parameters here.
        '''
        # TO-DO 3-1: Pass parameters to initialize model
        e = 2.7182818284
        size = config['vocab_size']
        import math

        self.config = config
        self.model = YourModel(
            inputSize = size,
            hiddenSize = 128,
            numLayer = 2,
            embededDim = 500
        ).to(config['device'])

    def forward(self, text):
        return self.model(text)
