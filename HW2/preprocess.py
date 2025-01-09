import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import re


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    # print(stop_word_list)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # return preprocessed_text
    text = re.sub(r'^a-zA-Z\s', '', text)
    
    
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    
    stopWords = set(stopwords.words('english'))
    
    tokens = [token.strip() for token in tokens]
    
    firstWord = tokens[0]
    tokens = tokens[1:]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    tokens = [firstWord] + [token for token in tokens if token not in stopWords]
    
    
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# print(preprocessing_function('Here is a dog.'))