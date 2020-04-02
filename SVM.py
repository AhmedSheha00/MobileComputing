#Support Vector Machine
from sklearn import svm
X = [[0, 0], [1, 1]]
#Input
y = [0, 1]
#Target
clf = svm.SVC()
# From svm in Sklearn , we can train a model for either Classifcation or Regression , so we use
#SVC from svm for a Classification Model
clf.fit(X, y)
#the fit function , used to train the model with the data
#it will adjust the weights according to the data to get a good accuracy , then use the model after
#training to predict
prediction1 = clf.predict([[0., 0.]])
prediction2 = clf.predict([[2., 2.]])
print("predictoin of first input:", prediction1)
print("predictoin of the second input:", prediction2)
#prediction of the target , when the inputs are [[0., 0.]] , [[2., 2.]] respectively
