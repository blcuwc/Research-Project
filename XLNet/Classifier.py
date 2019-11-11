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



def Load_data(dataset):


if __name__ == '__main__':
    dataset = sys.argv[1]
    Load_data(dataset)

