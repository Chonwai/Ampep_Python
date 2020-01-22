import sys
import numpy as np
from service import Utils as Utils
from service import GetFeature as GetFeature

fastaPath = sys.argv[1]
modelPath = sys.argv[2]

def main():
    GetFeature.getFeature(fastaPath, './data/test.tsv', 'CTDD')

    utils = Utils.Utils('Test')

    featureList, uselessY = utils.readFeature(
        "./data/test.tsv", 0)

    X = np.array(featureList)

    result = utils.predict(modelPath, X)

    # count = 0
    # for i in range(len(y)):
    #     if (result[i] == y[i]):
    #         count += 1
    
    print(result)

if __name__ == '__main__':
    main()
    