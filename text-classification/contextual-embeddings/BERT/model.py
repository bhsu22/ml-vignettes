"""Build a BERT classifier

Uses a BERT tf.Module.

Ref:
    https://towardsdatascience.com/fine-tuning-bert-with-keras-and-tf-module-ed24ea91cff2
    https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b
"""
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from bert.optimization import AdamWeightDecayOptimizer

from .bert_layer import BertLayer


def _majority_vote(chunk_predicted_probs, n_chunks, n_classes):
    votes = []

    for i in range(n_chunks):
        votes.append(np.argmax(chunk_predicted_probs[i], axis=1))

    votes = np.vstack(votes)
    predictions = np.zeros((votes.shape[1], n_classes))
    
    for sample in range(votes.shape[1]):
        counter = Counter(votes[:, sample])
        for k, v in counter.items():
            predictions[sample, k] = v/n_chunks

    return predictions


def _average_voting(chunk_predict_probs, n_chunks):
    probs = np.array(chunk_predict_probs)
    probs = np.sum(probs, axis=0)/n_chunks
    return probs

############################
#   MAIN METHODS
############################

def evaluate_model(bert_classifiers, bert_inputs, n_chunks, method="average"):
    input_ids, masks, segment_ids = bert_inputs

    chunk_predict_probs = []
    pooled_outputs = []

    for i in range(n_chunks):
        classifier = bert_classifiers[i]

        test_chk_ids = input_ids[i]
        test_chk_masks = masks[i]
        test_chk_segs = segment_ids[i]

        probs = classifier.predict([test_chk_ids, test_chk_masks, test_chk_segs])
        chunk_predict_probs.append(probs)

    if method == "average":
        return _average_voting(chunk_predict_probs, n_chunks)
    elif method == "majority":
        return _majority_vote(chunk_predict_probs, n_chunks)
    else:
        raise ValueError("method needs to be one of {average, majority}")


def create_model(bert_module_path, learning_rate=2e-5, max_seq_length=256, n_tune_layers=3, n_classes=20, optimizer="adam"):
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-6, amsgrad=True)
    adamW = AdamWeightDecayOptimizer(learning_rate, exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    input_ids = Input(shape=(max_seq_length,), name="input_ids")
    input_mask = Input(shape=(max_seq_length,), name="input_masks")
    input_segment = Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [input_ids, input_mask, input_segment]
    
    bert = BertLayer(bert_module_path,
                     seq_len=max_seq_length,
                     pooling='cls', # pooling='cls' returns pooled output, otherwise returns seqs
                     n_tune_layers=n_tune_layers,
                     use_layers=12,
                     trainable=True,
                     verbose=True
                    )
    dropout = Dropout(0.1)
    
    preds = Dense(n_classes, activation='softmax')(dropout(bert(bert_inputs)))
    model = Model(inputs=bert_inputs, outputs=preds)
    
    model.compile(loss='categorical_crossentropy',
              optimizer=adam if optimizer == "adam" else adamW,
              metrics=['acc'])
    model.summary()

    return model
