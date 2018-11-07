import bayes
listOPosts,listClasses=bayes.loadDataSet()
myVocabList=bayes.creatVocabList(listOPosts)
print(myVocabList)
print(listOPosts[0])
print(bayes.setOfWordsToVec(myVocabList,listOPosts[0]))