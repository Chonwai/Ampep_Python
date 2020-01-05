import sys
import numpy as np
from service import Utils as Utils
from service import GetFeature as GetFeature

feature = sys.argv[1]
modelPath = sys.argv[2]

def main():
    # GetFeature.getFeature('./data/testPos.fasta',
    #                       './data/testPos.tsv', feature)
    # GetFeature.getFeature('./data/testNav.fasta',
    #                       './data/testNav.tsv', feature)
    # GetFeature.getFeature('./data/trian_po_set3298_for_ampep_sever.fasta',
    #                       './data/trian_po_set3298_for_ampep_sever.tsv', feature)
    # GetFeature.getFeature('./data/trian_ne_set9894_for_ampep_sever.fasta',
    #                       './data/trian_ne_set9894_for_ampep_sever.tsv', feature)
    utils = Utils.Utils('Test')

    posArray, posY = utils.readFeature(
        "data/testPos.tsv", 1)
    negArray, negY = utils.readFeature(
        "data/testNav.tsv", 0)
    X = np.concatenate((posArray, negArray))
    y = np.concatenate((posY, negY))

    result = utils.predict(modelPath, X)

    count = 0
    for i in range(len(y)):
        if (result[i] == y[i]):
            count += 1
    
    print(str(count / len(result)))

if __name__ == '__main__':
    main()
    