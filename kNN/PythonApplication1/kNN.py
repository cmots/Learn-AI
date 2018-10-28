#coding:utf-8

from numpy import *
import operator

##给出训练数据以及对应的类别
def createDataSet():
    group = array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels = ['A','A','B','B']
    return group,labels

###通过KNN进行分类
def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0] ###获得dataSet第一维的长度
    ####计算欧式距离
    diff = tile(input,(dataSize,1)) - dataSet   ###将input重复，行上重复dataSize次（就会变成dataSize行的matrix），列上重复1次，矩阵宽度不变。目的是将输入矩阵和已知数据矩阵行数变成一样
    sqdiff = diff ** 2  ###diff是输入矩阵与已知矩阵的差，然后平方
    squareDist = sum(sqdiff,axis = 1)   ###行向量分别相加，从而得到新的一个行向量（就变成一个数组了），（当axis=0 会对列向量求和）
    dist = squareDist ** 0.5    ###开方得到欧几里得距离的一个数组
    
    ##对距离进行排序
    sortedDistIndex = argsort(dist)##argsort()根据元素的值从小到大对元素进行排序，返回下标

    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]   ##第i个原始数据的标签

        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1 ##字典classCount的键值是voteLabel，这步的操作是查找字典的某个元素保存的值（这里值代表标签个数），找到后对这个值+1（意思是这个标签又出现一次）

    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():    ##key就是标签，value就是这个字典中该标签保存的值（即出现次数）
        if value > maxCount:
            maxCount = value
            classes = key

    return classes