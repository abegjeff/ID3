# Most of the code examples were taken from http://stephenmonika.net/
# Which is the official website of the author of "Machine Learning: An Algorithmic Prospective"

#imports
import numpy as np
import math
import os
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import sys

'''Iris
 |     0      |      1      |      2      |     3     |     4    | 
   sepal length  sepal width  petal length petal width   class
'''

'''
 | 0 - 5.1   | 0 - 2.6     | 0 - 1.7  | 0 - 0.9 |
 | 5.1 - 6.5 | 2.6 - 3.3 |  1.7 - 5.0 | 0.9 - 7.0 |
 | 6.5 +     | 3.+       | 5+         | 7+
'''
#load Iris.data
def read_data(filename):
	fid = open(filename,"r")
	data = []
	d = []
	for line in fid.readlines():
		d.append(line.strip())
	for d1 in d:
		data.append(d1.split(","))
	fid.close()
	featureNames = data[0]
	featureNames = featureNames[2:]
	data = data[1:]
	classes = []
	for d in range(len(data)):
		classes.append(data[d][-1])
		data[d] = data[d][:-1]
	return data,classes,featureNames

####################################################################
def calc_entropy(p):
	if p != 0:
		return -p * math.log2(p)
	else:
		return 0

####################################################################
def calc_entropy_double(p1, p2):
	return calc_entropy(p1) + calc_entropy(p2)
####################################################################
def findPath(graph, start, end, pathSoFar):
	pathSoFar = pathSoFar + [start]
	if start == end:
		return pathSoFar
	if start not in graph:
		return None
	for node in graph[start]:
		if node not in pathSoFar:
			newpath = findPath(graph, node, end, pathSoFar)
			return newpath
	return None
#####################################################################
def calc_info_gain(data, classes, feature):
	gain = 0
	ggain = 0
	nData = len(data)
	# List the values that feature can take
	values = []
	for datapoint in data:
		if datapoint[feature] not in values:
			values.append(datapoint[feature])

	featureCounts = np.zeros(len(values))
	entropy = np.zeros(len(values))
	gini = np.zeros(len(values))
	valueIndex = 0
	# Find where those values appear in data[feature] and the corresponding class
	for value in values:
		dataIndex = 0
		newClasses = []
		for datapoint in data:			
			if datapoint[feature] == value:
				featureCounts[valueIndex] += 1
				newClasses.append(classes[dataIndex])
			dataIndex += 1
			

		#Get the values in newClasses #
		classValues = []
		for aclass in newClasses:
			if classValues.count(aclass) == 0:
				classValues.append(aclass)

		classCounts = np.zeros(len(classValues))
		classIndex = 0
		for classValue in classValues:
			for aclass in newClasses:
				if aclass == classValue:
					classCounts[classIndex] += 1
			classIndex += 1

		for classIndex in range(len(classValues)):
			entropy[valueIndex] += calc_entropy(float(classCounts[classIndex]) / sum(classCounts))
			gini[valueIndex] += (float(classCounts[classIndex])/np.sum(classCounts)) ** 2
		gain += float(featureCounts[valueIndex]) / nData * entropy[valueIndex]
		ggain += float(featureCounts[valueIndex])/nData * gini[valueIndex]
		valueIndex += 1
	return gain, 1-ggain
##############################################################################
def make_tree(data, classes, featureNames, maxLevel =- 1, Level = 0):
	nData = len(data)
	nFeatures = len(data[0])

	newClasses = []
	for aclass in classes:
		if newClasses.count(aclass) == 0:
			newClasses.append(aclass)

	frequency = np.zeros(len(newClasses))
	totalEntropy = 0
	totalGini = 0
	index = 0

	for aclass in newClasses:
		frequency[index] = classes.count(aclass)
		totalEntropy += calc_entropy(float(frequency[index])/nData)
		totalGini += (float(frequency[index])/nData) ** 2
		index += 1

	totalGini = 1 - totalGini

	default = classes[np.argmax(frequency)]

	if nData == 0 or nFeatures == 0 or (maxLevel >= 0 and level > maxLevel):
		# Have reaced an empty branch
		return default
	elif classes.count(classes[0]) == nData:
		# Only 1 class remains
		return classes[0]
	else:
		# Choose which feature is best
		gain = np.zeros(nFeatures)
		ggain = np.zeros(nFeatures)
		featureSet = range(nFeatures)
		for feature in featureSet:
			g,gg = calc_info_gain(data, classes, feature)
			gain[feature] = totalEntropy - g
			ggain[feature] = totalGini - gg

		bestFeature = np.argmax(gain)
		tree = {featureNames[bestFeature]:{}}
		# Find the possible feature values

		values = []
		for datapoint in data:
			if datapoint[feature] not in values:
				values.append(datapoint[bestFeature])

		for value in values:
			# Find the datapoints with each feature value
			newData = []
			newClasses = []
			index = 0
			for datapoint in data:
				if datapoint[bestFeature] == value:
					if bestFeature == 0:
						datapoint = datapoint[1:]
						newNames = featureNames[1:]
					elif bestFeature == nFeatures:
						datapoint = datapoint[:-1]
						newNames = featureNames[:-1]
					else:
						datapoint = datapoint[:bestFeature]
						datapoint.extend(datapoint[bestFeature+1:])
						newNames = featureNames[:bestFeature]
						newNames.extend(featureNames[bestFeature+1:])
					newData.append(datapoint)
					newClasses.append(classes[index])
				index += 1

		# Now recurse to the next level
		subtree = make_tree(newData, newClasses, newNames, maxLevel, Level)
		# And on returning, add the subtree on to the tree
		tree[featureNames[bestFeature]][value] = subtree
	return tree

##################################################################
def printTree(tree,name):
	if type(tree) == dict:
		keys = list(tree.keys())
		values = list(tree.values())
		print(name, keys[0])
		for item in values[0]:
			print(name, item)
			printTree(values[0][item], name + "\t")
	else:
		print(name, "\t->\t", tree)


def classify(tree,datapoint):
	#featureNames = ['setosa', 'versicolour', 'virginica']
	#featureNames = ['0', '1', '2']
	if type(tree) == type("string"):
		# Have reached a leaf
		return tree
	else:
		keys = list(tree.keys())
		a = keys[0]
		for i in range(len(featureNames)):
			if featureNames[i]==a:
				break
		
		try:
			t = tree[a][datapoint[i]]
			return classify(t,datapoint)
		except:
			print(sys.exc_info()[0])
			return None

def classifyAll(tree,data):
	results = []
	for i in range(len(data)):
		results.append(classify(tree,data[i]))
	return results


##########################################################################
# What runs the actual code
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
party,classes,features = read_data("iris.data") 

iris_data = party
iris_target = classes
iris_features = features
iris_attributes = attributes

the_tree = make_tree(party, classes, features)
"""
printTree(the_tree, ' ')
featureNames = features
print(classifyAll(the_tree,party))
for i in range(len(party)):
    classify(the_tree,party[i])
print("True Classes", classes)
"""
##########################################################################
# Existing implementation of ID3 from scikit-learn for Iris
iris = load_iris()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(iris.data, iris.target)

# Now display the answer

with open("iris.dot", 'w') as f:
	f = tree.export_graphviz(clf, out_file=f)

# Convert iris.dot into iris.pdf 
import os
os.system('dot -Tpdf iris.dot -o iris.pdf') # if the code fails here, then
											# install Graphviz on your machine


#delete iris.dot for next run
os.unlink('iris.dot')

########################################################
# Display section
# open iris.pdf // Do this at the end because the script
# keeps running while the pdf file is open
#os.system('iris.pdf')