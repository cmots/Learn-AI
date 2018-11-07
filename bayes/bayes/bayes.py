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