from numpy import *

#创建实验样本
def loadDataSet():
    postingList=[['my','dog','has','flea','problem','help','please'],
                ['maybe','not','take','him','to','dog','park','stupid'],
                ['my','dalmation','is','so','cute','I','love','him'],
                ['stop','posting','stupid','worthless','garbage'],
                ['mr','licks','ate','my','steak','how','to','stop','him'],
                ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]  #0是没有侮辱言论
    return postingList,classVec

def creatVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def setOfWordsToVec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in Vocabulary" %word)
    return returnVec

#训练朴素贝叶斯分类器
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)   #文档矩阵的行数
    numWords=len(trainMatrix[0])    #文档矩阵第0行的长度
    pAbusive=sum(trainCategory)/float(numTrainDocs) #因为骂人的行trainCategory为1，不骂人是0，求和就是骂人总行数
    #所以得到了骂人文本的概率
    p0Num=zeros(numWords)   #根据长度构造0向量
    p1Num=zeros(numWords)
    p0Denom=0.0
    p1Denom=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1: #这行骂人了
            p1Num+=trainMatrix[i]   #向量相加，是说骂人的行出现的词都要加到一个向量p1Num里
            p1Denom+=sum(trainMatrix[i])    #骂人的行内有多少单词的和
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=p1Num/p1Denom    #这是numpy的功能，意味着向量内每个值都除以了p1Denom，每个值代表了这个位置单词是骂人词汇的概率
    p0Vect=p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive
