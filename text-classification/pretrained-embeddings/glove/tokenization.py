import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import TransformerMixin


def train_test_split(X, y, val_fraction = 0.2, random_seed=0):
    indices = np.arange(X.shape[0])
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]
    
    num_validation_samples = int(val_fraction * X.shape[0])
    x_train, y_train = X[:-num_validation_samples], y[:-num_validation_samples]
    x_val, y_val = X[-num_validation_samples:], y[-num_validation_samples:]
    
    return x_train, y_train, x_val, y_val
    

class TextTokenizer(TransformerMixin):
    def __init__(self, max_num_words, max_sequence_length):
        super(TextTokenizer, self).__init__()
        
        self.max_num_words = max_num_words
        self.max_seq_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words = max_num_words)
        self.padder = pad_sequences
        
    def fit(self, X, y=None):
        """Fits a tokenizer """
        self.tokenizer.fit_on_texts(X)
        return self
    
    def transform(self, X, y=None):
        sequences = self.tokenizer.texts_to_sequences(X)
        data = self.padder(sequences, maxlen=self.max_seq_length)
        return data
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)