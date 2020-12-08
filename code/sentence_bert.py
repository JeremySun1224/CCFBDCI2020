# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/12/2 -*-

import warnings
import codecs
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from torch.nn import functional as F

os.chdir(path=r'E:\PycharmProjects\CCF')
warnings.filterwarnings(action='ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 利用sentence_bert直接对test进行标注
# tokenizer = AutoTokenizer.from_pretrained(r'pretrained_model\roberta-large-nli-mean-tokens\0_Transformer')
# model = AutoModel.from_pretrained(r'pretrained_model\roberta-large-nli-mean-tokens\0_Transformer')

tokenizer = BertTokenizer.from_pretrained(r'pretrained_model\roberta_clue\roberta_large_clue')
model = BertModel.from_pretrained(r'pretrained_model\roberta_clue\roberta_large_clue')
model = model.to(device)


# sentence = 'Who are you voting for in 2020?'
# labels = ['business', 'art & culture', 'politics']
# 读取测试集语料

def predict():
    out_list = []
    labels = ['finance & economics', 'house & property', 'decorate & furniture', 'education', 'science & technology', 'fashion', 'news & politics', 'game', 'entertainment', 'sports']
    # with codecs.open(filename=r'E:\PycharmProjects\CCF\data\colabdatatrans\test.txt', mode='r', encoding='utf-8') as f_test:
    with codecs.open(filename=r'E:\PycharmProjects\CCF\data\test_data.txt', mode='r', encoding='utf-8') as f_test:
        for idx, sentence in tqdm(enumerate(f_test)):
            inputs = tokenizer.batch_encode_plus([sentence] + labels, truncation=True, max_length=138, return_tensors='pt', pad_to_max_length=True).to(device)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            output = model(input_ids, attention_mask=attention_mask)[0]
            output = output.cpu()
            sentence_rep = output[:1].mean(dim=1)
            label_reps = output[1:].mean(dim=1)
            # now find the labels with the highest cosine similarities to the sentence
            similarities = F.cosine_similarity(sentence_rep, label_reps)
            closest = similarities.argsort(descending=True)
            # print(closest)
            # out_list.append(closest[0].numpy().tolist())
            out_list.append(closest[0].item())
            if idx % 100 == 0:
                print(f'result length: {len(out_list)}')
    return out_list


def write_out(results):
    with codecs.open(filename='./results/out.txt', mode='a', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(str(res) + '\n')


def main():
    results = predict()
    write_out(results=results)


if __name__ == '__main__':
    main()
