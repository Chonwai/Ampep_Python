import os
import numpy as np
import sys
import timeit
from service import GetFeature as GetFeature
from service import Trainer as RFTrainer
from service import Utils as Utils

feature = sys.argv[1]
model = sys.argv[2]
method = sys.argv[3]
fold = int(sys.argv[4])
trees = int(sys.argv[5])

def main():
    # GetFeature.getFeature('./data/trian_po_set3298_for_ampep_sever.fasta',
    #                       './data/trian_po_set3298_for_ampep_sever.tsv', feature)
    # GetFeature.getFeature('./data/trian_ne_set9894_for_ampep_sever.fasta',
    #                       './data/trian_ne_set9894_for_ampep_sever.tsv', feature)
    utils = Utils.Utils('Train')
    # posArray, posY = utils.readFeature(
    #     "data/trian_po_set3298_for_ampep_sever.tsv", 1)
    # negArray, negY = utils.readFeature(
    #     "data/trian_ne_set9894_for_ampep_sever.tsv", 0)
    posArray, posY = utils.readFeature(
        "data/matlab_pos.csv", 1)
    negArray, negY = utils.readFeature(
        "data/matlab_neg.csv", 0)
    X = np.concatenate((posArray, negArray))
    y = np.concatenate((posY, negY))

    trainer = RFTrainer.Trainer(X, y)
    trainer.training(fold, trees, model, method)

start = timeit.default_timer()
main()
stop = timeit.default_timer()
print('Time: ', stop - start)