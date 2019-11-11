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


def Set_input_embedding(base_model):
    # Len of the sentence must be the same as the training model
    # See model's 'max_position_embeddings' = 512
    max_len  = 64
    # With cased model, set do_lower_case = False
    tokenizer = XLNetTokenizer(vocab_file = base_model, do_lower_case = False)

def _3_fold_cv():
    return train_set, test_set

def _5_fold_cv():
    return train_set, test_set

def Load_data(dataset_dir, base_model):
    #read every dataset in dataset_dir into dataframe
    for dataset_name in os.listdir(dataset_dir):
        dataset_path = os.path.join(dataset_dir, dataset_name)
        df_data = pd.read_csv(dataset_path, sep="\t",encoding="utf-8",names=['texts','labels'])
        print (df_data.columns)
        df_data = df_data.drop(df_data['labels'] == 'NOT_LABELED')
        sentences = df_data['texts'].to_list()
        labels = df_data['labels'].to_list()



    # set input embedding using base xlnet model
    


if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    base_model = sys.argv[2]
    Load_data(dataset_dir, base_model)

