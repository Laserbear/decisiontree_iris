import numpy as np 
from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()
print "Feature names:", iris.feature_names
print "Classes:", iris.target_names


#print "Example %d: Label %s, features %s"%(i, iris.target[i], iris.data[i])

test_idx = [x for x in range(150) if x%5 == 0]

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

pred = clf.predict(test_data)
print test_target
print pred
print "Accuracy:", len([test_target[i] for i in range(len(test_target)) if test_target[i] == pred[i]])/float(len(test_target))