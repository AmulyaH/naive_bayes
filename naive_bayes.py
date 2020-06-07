import numpy as np
import csv
import math
import random


def prepareData():
    # Store the Yeast dataset CSV file names to load data   
    training_data = 'yeast_training.txt'
    testing_data = 'yeast_test.txt'

    # Load Yeast training dataset
    file1 = np.loadtxt(training_data)
    #data = file1.read()
    list_data = list(file1)
    for i in range(len(list_data)):
        list_data[i] = [float(x) for x in list_data[i]]  # Convert String to Float numbers
    train_data = list_data

    # Load  Yeast test dataset
    file2 = np.loadtxt(testing_data)
    #data1 = file2.read()
    list_data1 = list(file2)
    for i in range(len(list_data1)):
        list_data1[i] = [float(x) for x in list_data1[i]]  # Convert String to Float numbers
    test_data = list_data1

    return train_data, test_data

def mean(numbers):
    """Returns the mean of numbers"""
    return np.mean(numbers)

def stdev(numbers):
    """Returns the std_deviation of numbers"""
    return np.std(numbers)


def sigmoid(z):
    """Returns the sigmoid number"""
    return 1.0 / (1.0 + math.exp(-z))

    """Split training set by class value"""
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        row = dataset[i]
        if row[-1] not in separated:
            separated[row[-1]] = []
        separated[row[-1]].append(row)
    return separated

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	summaries.pop()
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for (classValue, instances) in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

#Calculate Gaussian Probability Density Function
def calculateProbability(x, mean, stdev):
    """Calculate probability using gaussian density function"""
    if (stdev == 0.0):
        if (x == mean):
            return 1.0
        else:
            return 0.0
    if (stdev < 0.01):
        stdev = 0.01
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return 1 / (math.sqrt(2 * math.pi) * stdev) * exponent

"""Calculate the class probability for input sample. Combine probability of each feature"""
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for (classValue, classSummaries) in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			(mean, stdev) = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

"""Compare probability for each class. Return the class label which has max probability."""
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	(bestLabel, bestProb) = (None, -1)
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

"""Get class label for each value in test set."""
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

#Get Accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))*100.0

def main():

    train_data, test_data = prepareData()

    summaries = summarizeByClass(train_data)

    predictions = getPredictions(summaries, test_data)

    accuracy = getAccuracy(test_data, predictions)
    print('Accuracy: {0}%'.format(accuracy))

if __name__== "__main__":
    main()

  