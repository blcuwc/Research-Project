#coding = utf-8
import torch
import os
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from pytorch_transformers import (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)

import pandas as pd
import math
import numpy as np
from sklearn.metrics import classification_report
import torch.nn.functional as F

import sys

disease_ids = ['DIS00', 'DIS01', 'DIS02']
category = ['factuality', 'polarity']


def Set_input_embedding(sentences, labels, vocabulary):
    # Len of the sentence must be the same as the training model
    # See model's 'max_position_embeddings' = 512
    max_len  = 64
    # With cased model, set do_lower_case = False
    tokenizer = XLNetTokenizer(vocab_file = vocabulary, do_lower_case = False)

    full_input_ids = []
    full_input_masks = []
    full_segment_ids = []

    SEG_ID_A   = 0
    SEG_ID_B   = 1
    SEG_ID_CLS = 2
    SEG_ID_SEP = 3
    SEG_ID_PAD = 4

    UNK_ID = tokenizer.encode("<unk>")[0]
    CLS_ID = tokenizer.encode("<cls>")[0]
    SEP_ID = tokenizer.encode("<sep>")[0]
    MASK_ID = tokenizer.encode("<mask>")[0]
    EOD_ID = tokenizer.encode("<eod>")[0]

    for i,sentence in enumerate(sentences):
        # Tokenize sentence to token id list
        tokens_a = tokenizer.encode(sentence)
    
        # Trim the len of text
        if(len(tokens_a) > max_len - 2):
            tokens_a = tokens_a[:max_len - 2]
        
        
        tokens = []
        segment_ids = []
    
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A)
        
        # Add <sep> token 
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)
    
    
        # Add <cls> token
        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)
    
        input_ids = tokens
    
        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length at fornt
        if len(input_ids) < max_len:
            delta_len = max_len - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len
    
        full_input_ids.append(input_ids)
        full_input_masks.append(input_mask)
        full_segment_ids.append(segment_ids)
    
        if 3 > i:
            print("No.:%d"%(i))
            print("sentence: %s"%(sentence))
            print("input_ids:%s"%(input_ids))
            print("attention_masks:%s"%(input_mask))
            print("segment_ids:%s"%(segment_ids))
            print("\n")
    return (full_input_ids, full_input_masks, full_segment_ids)

def _3_fold(Dict, base_model):
    name_list = Dict.keys()
    for name in name_list:
        test_inputs, test_tags, test_masks, test_segs = Dict[name]
        
    
    return train_set, test_set

def _5_fold(Dict, base_model):
    return train_set, test_set

def Load_data(dataset_dir, vocabulary):
    #read every dataset in dataset_dir into dataframe
    polarity_dict = {}
    factuality_dict = {}
    for dataset_name in os.listdir(dataset_dir):
        dataset_path = os.path.join(dataset_dir, dataset_name)
        df_data = pd.read_csv(dataset_path, sep="\t",encoding="utf-8",names=['texts','labels'])
        print (df_data.columns)
        df_data = df_data.drop(df_data['labels'] == 'NOT_LABELED')
        sentences = df_data['texts'].to_list()
        labels = df_data['labels'].to_list()

        # set input embedding using base xlnet model
        full_input_ids, full_input_masks, full_segment_ids = Set_input_embedding(sentences, labels, vocabulary)
        if 'polarity' in dataset_name:
            polarity_dict[dataset_name] = [full_input_ids, full_input_masks, full_segment_ids]
        if 'factuality' in dataset_name:
            factuality_dict[dataset_name] = [full_input_ids, full_input_masks, full_segment_ids]
    return polarity_dict, factuality_dict

def Classify(Dataset_dict, base_model, use_3=True, use_5=True):
    if use_3 == True:
        _3_fold(Dataset_dict, base_model)
    if use_5 == True:
        _5_fold(Dataset_dict, base_model)

if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    vocabulary = sys.argv[2]
    base_model = sys.argv[3]
    polarity_dict, factuality_dict = Load_data(dataset_dir, vocabulary)
    Classify(polarity_dict, base_model)
    Classify(factuality_dict, base_model)

