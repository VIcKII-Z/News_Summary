#!/usr/bin/python
#coding=utf-8
import sys
import numpy as np
import math

reload(sys)
sys.setdefaultencoding('utf-8')

def removeStopWord(words):
    stop_words = ['啊', '阿', '哎', '吧', '的', '等', '等等', '地',
        '在', '中', ',', '|', '?', '、', '。', '“', '”', '《', '》', '！', '，', '：', '；', '？']
    new_words = list()
    for w in words:
        if w not in stop_words:
            new_words.append(w)
    return new_words

def rmTfidfWords(words):
    rm_words = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '..', '...', '......', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '_', '__', '{', '|', '}', '~', '°', '±', '·', '×', 'é', 'α', 'β', '–', '—', '―', '‘', '’', '•', '…', '‰', '℃', 'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', '→', '∶', '≤', '≥', '⊙', '①', '②', '③', '④', '⑤', '■', '□', '▲', '○', '◎', '●', '★', '〇', '〈', '〉', '《', '》', '「', '」', '【', '】', '〔', '〕', '㎡', '一', '一一', '！', '％', '＋', '－', '．', '／', '：', '＝', '＞', '？', '［', '］', '｜', '～']
    new_words = list()
    for w in words:
        if w not in rm_words:
            new_words.append(w)
    return new_words

def euclideanDistance(x1,x2):
    return np.sqrt( np.sum((np.array(x1) - np.array(x2))**2) )

def cosSimilarityForLD(list1, list2):
    dict1 = dict()
    dict2 = dict()
    for i in list1:
        dict1[i[0]] = i[1]
    for i in list2:
        dict2[i[0]] = i[1]
    numerator = 0.0
    AA = 0.0
    BB = 0.0
    for k, v in dict1.items():
        if dict2.has_key(k):
            numerator = numerator + v * dict2[k]
        AA = AA + v * v
    for k, v in dict2.items():
        BB = BB + v * v
    if AA > 0.0 and BB > 0.0:
        sim = numerator / (math.sqrt(AA) * math.sqrt(BB))
        return sim

def cosSimilarityForList(list1, list2):
    if len(list1) != len(list2):
        return
    numerator = 0.0
    AA = 0.0
    BB = 0.0
    for i in range(0, len(list1)):
        numerator = numerator + list1[i] * list2[i]
        AA = AA + list1[i] * list1[i]
        BB = BB + list2[i] * list2[i]
    if AA > 0.0 and BB > 0.0:
        sim = numerator / (math.sqrt(AA) * math.sqrt(BB))
        return sim
