from numpy import *
import math
import sys
import operator

##通过举手表决的方式，确定传入的标签列表的label
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
        sortedClassCount=sorted(classCount.iteritems(),
                                key=operator.itemgetter(1),reverse=True)
        #根据标签出现次数做降序排列
    return sortedClassCount[0][0]

#计算信息熵
def calcEntropy(dataSet):
    numEntries=len(dataSet) #数据集的行数
    labelCounts={}
    for featVec in dataSet: #数据集每行
        currentLabel=featVec[-1]    #该行最后一个元素是label
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    Entropy=0.0
    for key in labelCounts:
        prob=float(labelCounts[key]/numEntries)
        Entropy-=prob*log(prob,2)
        #计算这个数据集的总信息熵
    return Entropy
#划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet: #fetVec是行向量
        if featVec[axis]==value:    #该行的这个维度满足条件
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) #这两行把该维度数据剔除
            retDataSet.append(reducedFeatVec)   #把剔除后子向量存到子集中
    return retDataSet
#寻找最好的特征值，然后划分数据集（成为子树）
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1   #数据集的特征数量（-1是因为不能包括label）
    baseEntropy=calcEntropy(dataSet)    #数据集的基信息熵
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]    #这行的意思是，从dataSet里获得每行example的第i项，就是说featList其实是数据集一个列向量构成的列表
        uniqueVals=set(featList)    #把featList重复的数据清除（就是说变成集合）
        newEntorpy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)    #保存第i维满足==value的行向量，构成子数据集
            prob=len(subDataSet)/float(len(dataSet))    #emmm估计是要算整个第i维的信息熵，就要各自乘以自己的占比再相加
            newEntorpy+=prob*calcEntropy(subDataSet)    #算出特征取值为value的信息熵，遍历直到value都算过一次
        infoGain=baseEntropy-newEntorpy #看剔除第i维后信息熵少了多少，就是信息增益
        if(infoGain>bestInfoGain):  #少了越多，也就是第i维的信息增益越多，第i维就是最好的分割特征
            bestInfoGain=infoGain
            bestFeature=i
        #然后重复，直至i从0到numFeatures，这样就能得到最好的分割特征维度
    return bestFeature

def creatTree(dataSet,labels):
    #labels是储存每个特征的名字的列表，而不是标签
    classList=[example[-1] for example in dataSet]  #数据集最后一维的列向量，就是标签列表
    if classList.count(classList[0])==len(classList):   #对标签列表里==第0个元素计数，如果等于标签列表数目，就是说标签列表里每个标签都一样
        return classList[0] #那么返回第0元素，就是整个数据集的标签
    if len(dataSet[0])==1:  #数据集第0行元素数目为1，就是说所有特征都被分割完了，这个时候数据集里还有不止一类标签
        return majorityCnt(classList)   #举手表决的方式选出标签返回
    bestFeat=chooseBestFeatureToSplit(dataSet)  #用奥卡姆剃刀原理分割数据集，得到分割的维度
    bestFeatLabel=labels[bestFeat]  #得到这个维度的特征名
    myTree={bestFeatLabel:{}}   #创造一个树的节点，key为这个特征名，树用字典实现
    del(labels[bestFeat])   #从特征名列表里删除这个特征名
    featValues=[example[bestFeat] for example in dataSet]   #得到这个维度的列向量
    uniqueVals=set(featValues)  #集合化
    for value in uniqueVals:
        subLabels=labels[:] #好像是python参数的问题吧，反正对算法没影响
        myTree[bestFeatLabel][value]=creatTree(splitDataSet\
            (dataSet,bestFeat,value),subLabels) #递归构造多个子树
    return myTree
