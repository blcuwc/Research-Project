#coding = utf-8
import torch
import os
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from sklearn.model_selection import train_test_split

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

def Fine_Tune(model, train_inputs, train_dataloader, batch_num ,device, num_train_optimization_steps, max_grad_norm):

    # True: fine tuning all the layers 
    # False: only fine tuning the classifier layers
    # Since XLNet in 'pytorch_transformer' did not contian classifier layers
    # FULL_FINETUNING = True need to set True
    FULL_FINETUNING = True

    if FULL_FINETUNING:
        # Fine tune model all layer parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        # Only fine tune classifier parameters
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    #Train model
    model.train()
    print("***** Running training *****")
    print("  Num examples = %d"%(len(train_inputs)))
    print("  Batch size = %d"%(batch_num))
    print("  Num steps = %d"%(num_train_optimization_steps))
    for _ in trange(epochs,desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_segs,b_labels = batch
        
            # forward pass
            outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
            loss, logits = outputs[:2]
            if n_gpu>1:
                # When multi gpu, average it
                loss = loss.mean()
        
            # backward pass
            loss.backward()
        
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        
            # update parameters
            optimizer.step()
            optimizer.zero_grad()
        
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
    return model

#def Save_model(model):

def accuracy(out, labels):
    outputs = np.argmax(out, axis = 1)
    return np.sum(outputs==labels)

def Evaluate_model(model, test_inputs, test_dataloader, batch_num):
    #evaluate model
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    y_true = []
    y_predict = []
    print("***** Running evaluation *****")
    
    print("  Num examples ={}".format(len(test_inputs)))
    print("  Batch size = {}".format(batch_num))
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_segs,b_labels = batch
        
        with torch.no_grad():
            outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
        tmp_eval_loss, logits = outputs[:2]
        
        # Get textclassification predict result
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)
#         print(tmp_eval_accuracy)
#         print(np.argmax(logits, axis=1))
#         print(label_ids)
            
        # Save predict and real label reuslt for analyze
        for predict in np.argmax(logits, axis=1):
            y_predict.append(predict)
            
        for real_result in label_ids.tolist():
            y_true.append(real_result)
     
            
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
       
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / len(val_inputs)
    loss = tr_loss/nb_tr_steps 
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'loss': loss}
    report = classification_report(y_pred=np.array(y_predict),y_true=np.array(y_true))

    # Save the report into file
    output_eval_file = os.path.join(xlnet_out_address, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print("  %s = %s"%(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))
        
        print(report)
        writer.write("\n\n")  
         
def _3_fold(Dict, base_model, tag2idx):
    name_list = Dict.keys()
    folds = []
    train_inputs = []
    train_tags = []
    train_masks = []
    train_segs = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_gpu = torch.cuda.device_count()

    #set batch num
    batch_num = 32

    #load model
    model = XLNetForSequenceClassification.from_pretrained(base_model, num_labels=len(tag2idx))

    # Set model to GPU,if you are using GPU machine
    model.to(device)

    # Add multi GPU support
    #if n_gpu >1:
        #model = torch.nn.DataParallel(model)

    # Set epoch and grad max num
    epochs = 5
    max_grad_norm = 1.0

    # Cacluate train optimiazaion num
    num_train_optimization_steps = int( math.ceil(len(tr_inputs) / batch_num) / 1) * epochs

    for name in name_list:
        test_inputs, test_tags, test_masks, test_segs = Dict[name]
        for _name in name_list:
            if _name != name:
                train_inputs.extend(Dict[_name][0])
                train_tags.extend(Dict[_name][1])
                train_masks.extend(Dict[_name][2])
                train_segs.extend(Dict[_name][3])
        
        train_inputs = torch.tensor(train_inputs)
        test_inputs = torch.tensor(test_inputs)
        train_tags = torch.tensor(train_tags)
        test_tags = torch.tensor(test_tags)
        train_masks = torch.tensor(train_masks)
        test_masks = torch.tensor(test_masks)
        train_segs = torch.tensor(train_segs)
        test_segs = torch.tensor(test_segs)

        train_data = TensorDataset(train_inputs, train_masks, train_segs, train_tags)
        train_sample = RansomSampler(train_data)
        # Drop last can make batch training better for the last one
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_num, drop_last = True)
        
        test_data = TensorDataset(test_inputs, test_masks, test_segs, test_tags)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=valid_sampler, batch_size=batch_num)

        folds.append([train_inputs, train_dataloader, test_inputs, test_dataloader])

        train_inputs = []
        train_tags = []
        train_masks = []
        train_segs = []

    for fold in folds:
        train_inputs, train_dataloader, test_inputs, test_dataloader = fold
        #Fine-tune model        
        model = Fine_Tune(model, train_inputs, train_dataloader, batch_num, device, num_train_optimization_steps, max_grad_norm)

        #Save model
        #Save_model(model)
    
        Evaluate_model(model, test_inputs, test_dataloader, batch_num)


def _5_fold(Dict, base_model, tag2idx):
    pass

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

def Classify(Dataset_dict, base_model, tag2idx, use_3=True, use_5=True):
    if use_3 == True:
        _3_fold(Dataset_dict, base_model, tag2idx)
    if use_5 == True:
        _5_fold(Dataset_dict, base_model, tag2idx)

if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    vocabulary = sys.argv[2]
    base_model = sys.argv[3]
    polarity_dict, factuality_dict = Load_data(dataset_dir, vocabulary)
    p_tag2idx = {'POSITIVE':0, 'NEUTRAL':1, 'NEGATIVE':2}
    f_tag2idx = {'EXPERIENCE':0, 'OPINION':1, 'FACT':2}
    Classify(polarity_dict, base_model, p_tag2idx)
    Classify(factuality_dict, base_model, f_tag2idx)

