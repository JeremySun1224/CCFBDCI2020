# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/11/12 -*-

import os
import csv
import copy
import random
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn.utils import clip_grad_norm_
from sklearn.utils import shuffle as reset
from tqdm import tqdm
import transformers
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False

classes2idx = {'财经': 0, '房产': 1, '家居': 2, '教育': 3, '科技': 4, '时尚': 5, '时政': 6, '游戏': 7, '娱乐': 8, '体育': 9}
idx2classes = {0: '财经', 1: '房产', 2: '家居', 3: '教育', 4: '科技', 5: '时尚', 6: '时政', 7: '游戏', 8: '娱乐', 9: '体育'}

MODEL_NAME = '../pretrained_model/roberta_base_myself'
OUTPUT_MODEL = '../models/roberta_base_myself.pth'

SEED = 1224
TRAIN = True
BATCH_SIZE = 4
EPOCHS = 5
HIDDEN_DIM = 256
DROPOUT = 0.3
BIDIRECTIONAL = True

best_score = 0.


class CustomDataset(Data.Dataset):
    def __init__(self, data, max_len, with_labels=True, model_name=MODEL_NAME):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent = str(self.data.loc[index, 'content'])
        encoded_pair = self.tokenizer(sent, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with 0 for padded values and 1 for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with 0 for the 1st sentence tokens and 1 for the 2nd sentence tokens

        if self.with_labels:
            label = self.data.loc[index, 'class_label']
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


class CLSModel(nn.Module):
    def __init__(self, model_name, hidden_dim, num_classes, dropout, bidirectional=BIDIRECTIONAL, freeze_bert=False):
        super(CLSModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(input_size=embedding_dim * 4, hidden_size=hidden_dim, batch_first=True, dropout=dropout, bidirectional=bidirectional)  # [batch_size, seq_len, input_size]
        # self.linear = nn.Linear(in_features=hidden_dim * 2, out_features=num_classes)
        self.linear = nn.Sequential(
            nn.Linear(in_features=hidden_dim * 2, out_features=512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        self.dropout = nn.Dropout(p=dropout)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attn_masks, token_type_ids):
        embedding = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attn_masks)  # [batch_size, seq_len, hidden_size]
        hidden_states = torch.cat(tuple([embedding.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1)  # [batch_size, seq_len, hidden_size * 4]
        first_hidden_states = hidden_states[:, 0, :]  # [batch_size, hidden_size * 4]
        first_hidden_states = first_hidden_states.unsqueeze(dim=1)  # [batch_size, 1, hidden_size * 4]
        # first_hidden_states.permute(1, 0, 2)  # [1, batch_size, hidden_size * 4]
        _, hidden = self.rnn(first_hidden_states)  # [num_layers * num_directions, batch_size, hidden_dim]
        hidden = self.dropout(torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1) if self.rnn.bidirectional else hidden[-1, :, :])  # torch.cat([batch_size, hidden_dim], [batch_size, hidden_dim]) --> [batch_size, hidden_dim * 2]
        logits = self.linear(hidden)
        return logits


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_data(filename, classes2idx, with_labels=True):
    data = pd.read_csv(filename, encoding='utf-8')
    print(len(data))
    if with_labels:
        data = data.replace({'class_label': classes2idx})
    return data


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def save(model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, OUTPUT_MODEL)
    print('\nThe best model has been saved.')


def train_test_split(data_df, test_size=0.2, shuffle=True, random_state=None):
    if shuffle:
        data_df = reset(data_df, random_state=random_state)

    train = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test = data_df[: int(len(data_df) * test_size)].reset_index(drop=True)

    return train, test


def train_eval(model, criterion, optimizer, train_loader, val_loader, epochs):
    # checkpoint = torch.load(output_model, map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    # total_steps = len(train_loader) * epochs

    print('========== Training ==========')

    for epoch in range(epochs):
        model.train()
        print(f'Epoch {epoch + 1}')
        for i, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            logits = model(batch[0], batch[1], batch[2])
            loss = criterion(logits, batch[3])
            # if i % 10 == 0:
            # print(i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # scheduler.step()

            if i % 200 == 0:
                evaluate(model, optimizer, val_loader)


def evaluate(model, optimizer, validation_dataloader):
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0., 0., 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    print('\nValidation Accuracy: {}'.format(eval_accuracy / nb_eval_steps))

    global best_score

    if best_score < eval_accuracy / nb_eval_steps:
        best_score = eval_accuracy / nb_eval_steps
        save(model, optimizer)

    # model.train()


def test(model, dataloader, with_labels=False):
    checkpoint = torch.load(OUTPUT_MODEL, map_location='cpu')  # load model

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print('========== Testing ==========')

    pred_label = []
    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)

    pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('../results/pred.csv', header=['class_label'], encoding='utf-8')

    rel_dict = {'财经': '高风险', '时政': '高风险', '房产': '中风险', '科技': '中风险', '教育': '低风险', '时尚': '低风险', '游戏': '低风险', '家居': '可公开', '体育': '可公开', '娱乐': '可公开'}

    with open('../results/pred.csv', encoding='utf-8') as f:
        rows = [row for row in csv.reader(f)]
        rows = np.array(rows[1:])
        label_list = [label for _, label in rows]  # label list
        final_col = []
        for i in label_list:
            final_col.append(rel_dict[idx2classes[int(i)]])

        data = pd.read_csv('../results/pred.csv')
        data['final'] = final_col
        data = data.replace({'class_label': idx2classes})
        data.to_csv('../results/result.csv', index=False, encoding='utf_8_sig', header=['id', 'class_label', 'rank_label'])

    print('Test completed')


if __name__ == '__main__':
    set_seed(1224)

    data_df = process_data(filename='../data/train/all_labeled_data.csv', classes2idx=classes2idx, with_labels=True)
    train_df, val_df = train_test_split(data_df=data_df, test_size=0.2, shuffle=True, random_state=1)

    print('Reading training data...')
    train_set = CustomDataset(data=train_df, max_len=512, model_name=MODEL_NAME)
    train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=5, shuffle=True)

    print('Reading validation data...')
    val_set = CustomDataset(data=val_df, max_len=512, model_name=MODEL_NAME)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, num_workers=5, shuffle=True)

    model = CLSModel(model_name=MODEL_NAME, hidden_dim=HIDDEN_DIM, num_classes=len(classes2idx), dropout=DROPOUT, bidirectional=BIDIRECTIONAL, freeze_bert=False)
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    train_eval(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, epochs=EPOCHS)

    print('Reading test data...')
    test_df = process_data(filename='../data/test_data_cleaned.csv', classes2idx=classes2idx, with_labels=False)
    test_set = CustomDataset(data=test_df, max_len=512, with_labels=False, model_name=MODEL_NAME)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=5, shuffle=False)

    test(model=model, dataloader=test_loader, with_labels=False)
