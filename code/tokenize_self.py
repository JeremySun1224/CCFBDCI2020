# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/11/26 -*-

import jieba
import codecs
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


# tokenizer = BertTokenizer.from_pretrained('../pretrained_model/wobert_base')


def demo(text, do_basic_tokenize=True, never_split=None):
    bert_tokenizer = BertTokenizer.from_pretrained('../pretrained_model/wobert_base', do_basic_tokenize=do_basic_tokenize, never_split=never_split)
    res = bert_tokenizer(text)
    input_ids = res["input_ids"]
    input_tokens = []
    for id in input_ids:
        token = bert_tokenizer._convert_id_to_token(id)
        input_tokens.append(token)
    print(input_tokens)


def tokenize_(text):
    tokenize_text = ' '.join(jieba.cut(text))
    return tokenize_text


def read_(file_in, file_out):
    with codecs.open(filename=file_in, mode='r', encoding='utf-8') as f_in:
        for ctx in f_in.readlines():
            tokenize_ctx = tokenize_(text=ctx)
            with codecs.open(file_out, mode='a', encoding='utf-8') as f_out:
                f_out.write(tokenize_ctx)
    print('finished')


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _is_chinese_char(cp):
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True

    return False


if __name__ == '__main__':
    # sentence = '我爱北京天安门。'
    # demo(text=sentence, do_basic_tokenize=False)
    # sentence = "我 爱 北京 天安门 ， 天安门 上 太阳升 。 人民 领袖 毛主席 ，指引 我们 向前进 。 "
    # print(f'sentence: {sentence}')
    #
    # with codecs.open(filename='./测试分词.txt', mode='r', encoding='utf-8') as f:
    #     for sentence in f.readlines():
    #         tokenize_sentence = tokenize_(text=sentence)
    #         # print(f'tokenize_sentence: {tokenize_sentence}')
    #         demo(text=sentence, do_basic_tokenize=False)
    #         print('===' * 15)
    # demo(text=sentence, do_basic_tokenize=False)
    # context = read_(file_in='./测试.txt', file_out='./测试分词.txt')
    # print(_tokenize_chinese_chars(text=sentence))
    # 分词文件
    # read_(file_in='../data/train/labeled_data.txt', file_out='../data/train/labeled_data_tokenized.txt')
    # read_(file_in='../data/train/unlabeled_data.txt', file_out='../data/train/unlabeled_data_tokenized.txt')
    # read_(file_in='../data/test_data.txt', file_out='../data/test_data_tokenized.txt')
    # read_(file_in='../data/train/all_content.txt', file_out='../data/train/all_content_tokenized.txt')
    test_sentence = '将设计融于人性，将家居带入悠闲自在的情境。'
    print(' '.join(jieba.cut(sentence=test_sentence)))