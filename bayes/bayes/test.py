import bayes
listOPosts,listClasses=bayes.loadDataSet()
myVocabList=bayes.creatVocabList(listOPosts)
trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWordsToVec(myVocabList,postinDoc))
p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
#print(p1V)
#print(myVocabList)
for i in range(len(p1V)):
    print("%s %lf" % (myVocabList[i],p1V[i]))