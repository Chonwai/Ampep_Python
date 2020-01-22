import os
import numpy as np
import sys
import timeit
from service import GetFeature as GetFeature
from service import Trainer as Trainer
from service import Utils as Utils

feature = sys.argv[1]
model = sys.argv[2]
trees = int(sys.argv[3])

def main():
    GetFeature.getFeature('./data/trian_po_set3298_for_ampep_sever.fasta',
                          './data/trian_po_set3298_for_ampep_sever.tsv', feature)
    GetFeature.getFeature('./data/trian_ne_set9894_for_ampep_sever.fasta',
                          './data/trian_ne_set9894_for_ampep_sever.tsv', feature)
    utils = Utils.Utils('Train')
    posArray, posY = utils.readFeature(
        "data/trian_po_set3298_for_ampep_sever.tsv", 1)
    negArray, negY = utils.readFeature(
        "data/trian_ne_set9894_for_ampep_sever.tsv", 0)
    X = np.concatenate((posArray, negArray))
    y = np.concatenate((posY, negY))

    trainer = Trainer.Trainer(X, y)
    trainer.trainingModel(trees, model)

start = timeit.default_timer()
main()
stop = timeit.default_timer()
print('Time: ', stop - start)