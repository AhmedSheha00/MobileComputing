from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
iris = load_iris()
#Loading the IRIS Dataset
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
#Setting the Classifier of the Decision Tree (model)
decision_tree = decision_tree.fit(iris.data, iris.target)
#the fit function , used to train the model with the data
#it will adjust the weights according to the data to get a good accuracy , then use the model after
#training to predict
r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r) #the output , first checks if the width of petal <= 0.8 , then if it's > 0.8 then <= 1.75 , then if it's > 1.75
#then output of the classification model is CLASS 2
