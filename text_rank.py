# #!/usr/bin/python
# # coding=utf-8
#
# import numpy as np
# import pandas as pd
# import nltk
# from nltk.tokenize import sent_tokenize
#
# import re
#
# with open('tennis_articles_v4.csv','r',encoding='utf-8') as f:
#     df=pd.read_csv(f)
# print(df.head())
# sentences = []
# for s in df['article_text']:
#   sentences.append(sent_tokenize(s))
#
#
#
# # flatten the list
# sentences = [y for x in sentences for y in x]
#
#
# # remove punctuations, numbers and special characters
# clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
#
# # make alphabets lowercase
# clean_sentences = [s.lower() for s in clean_sentences]
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# word_embeddings = {}
# f = open('glove.6B/glove.6B.100d.txt', encoding='utf-8')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     word_embeddings[word] = coefs
# f.close()
#
# sentence_vectors = []
# for i in clean_sentences:
#   if len(i) != 0:
#     v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
#   else:
#     v = np.zeros((100,))
#   sentence_vectors.append(v)
# # similarity matrix
# sim_mat = np.zeros([len(sentences), len(sentences)])
#
#
# from sklearn.metrics.pairwise import cosine_similarity
#
#
#
# for i in range(len(sentences)):
#   for j in range(len(sentences)):
#     if i != j:
#       sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
#
#
#
# import networkx as nx
#
# nx_graph = nx.from_numpy_array(sim_mat)
# scores = nx.pagerank(nx_graph)
#
#
#
# ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
#
#
#
# # Specify number of sentences to form the summary
# sn = 10
#
# # Generate summary
# for i in range(sn):
#   print(ranked_sentences[i][1])
#
#

import sys
import jieba
import helper
import re
from gensim import corpora, models, similarities
import imp
imp.reload(sys)
sys.setdefaultencoding('utf-8')

min_sentence_words_num = 5
min_sentence_len = 11
damping_factor = 0.85


class TextRank(object):
    def __init__(self, model, model_dict):
        self.tfidf_model = models.TfidfModel.load(model)
        self.dictionary = corpora.Dictionary.load(model_dict)

    def getTfidfVec(self, text):
        return self.tfidf_model[self.dictionary.doc2bow(list(jieba.cut(text)))]

    def getTextSentences(self, text):
        parts = re.split('。|！|？|\n', text)
        sentences = list()
        for p in parts:
            if len(p.strip().decode('utf-8')) <= min_sentence_len:
                continue
            sentences.append(p)
        return sentences

    def getSimMatrix(self, sent_vecs):
        sim_matrix = list()
        size = len(sent_vecs)
        for i in range(0, size):
            list_tmp = list()
            for j in range(0, size):
                list_tmp.append(-1)
            sim_matrix.append(list_tmp)
        total_sim = 0.0
        cnt = 0
        for i in range(0, size):
            for j in range(i + 1, size):
                sim = helper.cosSimilarityForLD(sent_vecs[i], sent_vecs[j])
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
                total_sim = total_sim + sim
                cnt = cnt + 1
        avg_sim = 0.0
        if cnt > 0:
            avg_sim = total_sim / cnt
        '''
        for i in range(0, size):
            for j in range(0, size):
                if sim_matrix[i][j] <= avg_sim:
                    sim_matrix[i][j] = -1
        '''
        out_vecs = list()
        for i in range(0, size):
            out_tmp = 0.0
            for j in range(0, size):
                if sim_matrix[i][j] > 0.0:
                    out_tmp = out_tmp + sim_matrix[i][j]
            out_vecs.append(out_tmp)
        return (sim_matrix, out_vecs)

    def summarize(self, text, summary_len):
        text = text.strip().lower()
        sentences = self.getTextSentences(text)
        sent_vecs = list()
        for i in range(0, len(sentences)):
            sent_vecs.append(self.getTfidfVec(sentences[i]))
        sim_matrix, out_vecs = self.getSimMatrix(sent_vecs)
        sent_scores = list()
        for i in range(0, len(sentences)):
            sent_scores.append(1.0)
        epochs = 20
        for e in range(0, epochs):
            scores_tmp = list()
            for i in range(0, len(sentences)):
                score_i = 1 - damping_factor
                for j in range(0, len(sentences)):
                    if sim_matrix[i][j] > 0.0:
                        score_i = score_i + damping_factor * sim_matrix[i][j] * sent_scores[j] / out_vecs[j]
                scores_tmp.append(score_i)
            sent_scores = scores_tmp
        score_dict = dict()
        for i in range(0, len(sentences)):
            score_dict[i] = sent_scores[i]
        sorted_sent_score = sorted(score_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        selected_sent_ids = list()
        selected_summary_len = 0
        for item in sorted_sent_score:
            if item[0] in selected_sent_ids:
                continue
            selected_sent_ids.append(item[0])
            selected_summary_len = selected_summary_len + len(sentences[item[0]].decode('utf-8'))
            if selected_summary_len >= summary_len:
                break
        selected_sent_ids.sort()
        summary = ''
        for i in selected_sent_ids:
            summary = summary + sentences[i] + '。'
        summary = summary.replace('。。', '。').replace('！。', '！').replace('？。', '？')
        return summary


def test():
    text_rank = TextRank('../../model/news_train_clean_5_0.4_500000_tfidf',
                         '../../model/news_train_clean_5_0.4_500000.dict')
    text = '中国证券网讯 思美传媒11日早间调整公司资产重组方案。根据原方案，公司拟以发行股份及支付现金的方式购买北京爱德康赛广告有限公司（以下简称“爱德康赛”）100%股权，其中，交易对价的50%以发行股份的方式支付，交易对价的50%以现金方>式支付，同时非公开发行股份募集配套资金。以2015年3月31日为基准日，爱德康赛100%股权的评估值为35,530.95万元。然而，以2015年9月30日为基准日，爱德康赛100%股权的评估值降为29,324.61万元。鉴于搜索引擎营销行业市场环境因素变化，爱德康赛未来预测经营业绩和估值发生变化，公司决定对方案进行调整。经协商确定交易价格为29,000万元，其中，交易对价的50%以发行股份的方式支付，交易对价的50%以现金方式支付。同时，募集配套资金总额预计不超过28,999.98万元。调整后，交易对方承诺爱德康赛>于2015年度、2016年度、2017年度、2018年度、2019年度应实现的年度经审核税后净利润应分别不低于1,400万元、2,700万元、3,510万元、4,563万元、5,019.3万元。本次调整后募集配套资金总额不超过28,999.9843万元，其中14,500.00万元用于支付本次交易>的现金对价，8,250.18万元用于补充公司流动资金，剩余部分用于扩大媒介代理业务规模和支付本次相关发行费用。思美传媒同日>公告，拟以自有资金人民币5000万元投资浙江文化产业成长基金（有限合伙）（以下简称“浙江文产基金。浙江文产基金目标募集规模拟定为50亿元。其中，思美传媒认缴基金份额拟定为人民币5000万元，其他有限合伙人认缴剩余部分。本基金管理人拟定为管理>团队与浙江省文化促进会合资成立的有限责任公司，受托管理和运作基金，履行对基金的实际管理职能。'
    print
    text_rank.summarize(text, len(text.decode('utf-8')) * 0.2)


if __name__ == '__main__':
    test()
a