"""Builds a 1D Convnet with MaxPooling using GLoVe embeddings.

Ref: https://keras.io/examples/pretrained_word_embeddings/
"""
import numpy as np

from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant


def get_embedding_index(filepath):
    embeddings_index = {}
    with open(filepath) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    return embeddings_index


def get_embedding_matrix(word2idx, embedding_idx, max_num_words, embedding_dim):
    """Get glove embedding matrix."""
    # prepare embedding matrix
    num_words = min(max_num_words, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word2idx.items():
        if i >= max_num_words:
            continue
        embedding_vector = embedding_idx.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_model(embedding_matrix, max_seq_len, activation='relu', n_classes=20):
    """Build a 1D convnet with global maxpooling."""
    n_words, emb_dim = embedding_matrix.shape
    
    embedding_layer = Embedding(n_words,
                                emb_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_seq_len,
                                trainable=False)
    
    
    sequence_input = Input(shape=(max_seq_len,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation=activation)(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation=activation)(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation=activation)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation=activation)(x)
    preds = Dense(n_classes, activation='softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
    return model