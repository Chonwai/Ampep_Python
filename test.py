import sys
import numpy as np
from service import Utils as Utils
from service import GetFeature as GetFeature

feature = sys.argv[1]
fastaPath = sys.argv[2]
modelPath = sys.argv[3]

def main():
    GetFeature.getFeature(fastaPath, './data/test.tsv', feature)

    utils = Utils.Utils('Test')

    featureList, uselessY = utils.readFeature(
        "./data/test.tsv", 0)

    X = np.array(featureList)

    result = utils.predict(modelPath, X)
    
    print(result)

if __name__ == '__main__':
    main()
