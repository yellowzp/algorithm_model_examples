#! /usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
import jieba
import pdb
import re
import datetime


def cur_text_data():
    f = open("C:\\Users\\Administrator\\data\\zhwiki\\zhwiki_2017_03.clean", "r")
    output = open("zhwiki_2017_03_segs", "w")
    r = re.compile(ur"[\t\s+\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？、~@#￥%……&*（）()]+")
    for line in f:
        mid_text = re.sub(r, "", line.strip().decode("utf8"))
        segs = " ".join(jieba.cut(mid_text))
        output.write(segs.encode("utf8") + "\n")
    f.close()
    output.close()


def train(text_path):
    """
    训练100k条数据20分钟

    :param text_path:
    :return:
    """
    start_t = datetime.datetime.now()
    print start_t
    model = Word2Vec(LineSentence(text_path), size=100, min_count=10, hs=0, window=5, sg=1)
    end_t = datetime.datetime.now()
    print "train cost time ", end_t - start_t
    model.save("w2v_%s" % text_path)


def test(model_path):
    model = Word2Vec.load(model_path)
    # # 打印词向量
    # print model["数学".decode("utf8")]
    # # 词之间相似度
    # print model.wv.similarity(u"时期", u"种类")
    # # 找相关词
    # res = model.wv.most_similar(u"戏剧")
    # for tup in res:
    #     print tup[0], tup[1]
    print model.wv.similarity(u"笔记本", u"台式")
    print model.wv.similarity(u"笔记本", u"笔记本")



if __name__ == "__main__":
    # cur_text_data()
    # train("zhwiki_2017_03_segs_100k")
    test("w2v_zhwiki_2017_03_segs_100k")
