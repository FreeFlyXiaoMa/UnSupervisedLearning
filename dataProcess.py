import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn
from os import path

import jieba
'''
#创建停用词列表
def stopwordslist():
    stopwords=[line.strip() for line in open('stopwords.txt'
                                             ,'rb')
                                    .readlines()]
    return stopwords

#对句子进行中文分词
def seg_depart(sentence):
    #对文档中的每一行进行中文分词
    print('正在分词')
    sentence_depart=jieba.cut(sentence.strip())
    #创建一个停用词列表
    stopwords=stopwordslist()
    #输出结果
    outstr=''
    #去除停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr +=word
                outstr +=" "
    return outstr

#给出文档路径
filename='training.csv'

outfilename='out.txt'
inputs=open(filename,'rb')
outputs=open(outfilename,'wb')

#将输出结果写入out.txt中
for line in inputs:
    line_seg=seg_depart(line)
    line_seg.encode('UTF-8')
    line_seg_object=line_seg+'\n'
    line_seg_object.encode('UTF-8')
    outputs.write(line_seg_object)
    print("------------正在分词和停用词-------------")

outputs.close()
inputs.close()

print("删除停用词和分词成功")'''
'''
import jieba.analyse
stopwords=[]
for word in open('stopwords.txt','rb'):
    stopwords.append(word.strip())
    article=open('training.csv','rb').read()
    words=jieba.cut(article,cut_all=False)
    stayed_line=""
    for word in words:
        if word not in stopwords:
            stayed_line +=word+" "
    #print(stayed_line)
    w=open('out.txt','wb')
    w.write(stayed_line.encode('utf-8'))'''

from sklearn.feature_extraction.text import TfidfVectorizer

#读入停用词
stopWordFile='stopwords.txt'
fr=open(stopWordFile,'rb')
words=fr.read()
#print(words)
stopWords=jieba.cut(words,cut_all=True)
stopWords=list(stopWords)

#定义去掉停用词方法
def wordsCut(words):
    result=jieba.cut(words)
    newWords=[]
    for s in result:
        if s not in stopWords:
            newWords.append(s)
    return  ' '.join(newWords)

#每一行操作
data=pd.read_csv('training.csv',error_bad_lines=False)
#print(data.shape)
#print(data['id'])
data['content']=data['content'].apply(lambda x:wordsCut(x))
print(data['content'])
#tfidf实例
cv=TfidfVectorizer()
cv_fit=cv.fit_transform(data['content'])
x=cv_fit.toarray()
#print(x)
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

#定义聚类模型
def K_cluster_analysis(K,X):
    print('K-means begin with cluster:{}'.format(K))

    mb_kmeans=MiniBatchKMeans(n_clusters=K)
    y_pred=mb_kmeans.fit_predict(X)

    #采用无参考默认的评价指标：轮廓稀疏Silhoutte Coefficient和
    #Calinski-Sarabasz Index
    CH_score=metrics.calinski_harabaz_score(X,y_pred)

    print('CH_score:{}'.format(CH_score))
    return CH_score

#设置超参数（聚类数目）搜索范围
Ks=range(2,6,1)
CH_scores=[]
for K in Ks:
    ch=K_cluster_analysis(K,x)
    CH_scores.append(ch)



#最佳超参数
index=np.unravel_index(np.argmax(CH_scores,axis=None),len(CH_scores))
Best_K=Ks[index[0]]
print(Best_K)

#绘制不同K对应的聚类性能，找到最佳模型/参数
import matplotlib.pyplot as plt
plt.plot(Ks,np.array(CH_scores),'b--',label='CH_score')
plt.show()

