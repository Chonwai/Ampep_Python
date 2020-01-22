import numpy as np
import pickle

class Utils():
    def __init__(self, name = ''):
        self.name = name

    def readFeature(self, path = '', y = 0):
        f = open(path, "r")
        items = f.readlines()
        newArray = []
        for item in items:
            tempArray = np.array(item.rstrip().split('\t'))
            newArray.append(tempArray[1:])
        newArray = np.array(newArray[1:])
        labelArray = []
        for i in range(len(newArray)):
            labelArray.append(y)
            i = i + 1
        return newArray, labelArray

    def predict(self, path, X):
        with open(path, 'rb') as f:
            clf = pickle.load(f)
            result = clf.predict(X)
            return result
                
        