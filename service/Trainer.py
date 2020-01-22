import sklearn
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, RandomTreesEmbedding, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn import svm
import pickle
import statistics
import math


class Trainer():
    def __init__(self, X=[], y=[]):
        self.X = X
        self.y = y
        self.trees = 100
        self.nJobs = 4

    def trainingCV(self, fold=10, trees=100, model="RandomForestClassifier", method="ShuffleSplit", step=30):
        self.trees = trees
        self.writeCSVHeader(method, fold, model)
        for i in range(1, step + 1):
            trees = self.trees + (100 * (i - 1))
            clf = self.modelSelector(model, trees)
            cv = self.cvSelector(method, fold)
            sn, sp, accuracy, mcc, rocAuc, k = self.fitCV(clf, cv)
            print("Finished Training Model " + str(i) + " Times with " +
                  str(fold) + " Fold and " + str(trees) + " Trees!")
            self.writeCSVContent(sn, sp, accuracy, mcc, rocAuc, k, i, method, fold, model, trees)

    def trainingModel(self, trees=100, model="RandomForestClassifier"):
        self.trees = trees
        clf = self.modelSelector(model, trees)
        self.fitModel(clf, model)
        print("Finished Training " + model + " Model and " + str(trees) + " Trees! \n")

    def fitCV(self, clf, cv):
        snList = []
        spList = []
        accuracyList = []
        mccList = []
        rocAucList = []
        kList = []
        for train_index, test_index in cv.split(self.X, self.y):
            train_X, test_X = self.X[train_index], self.X[test_index]
            train_y, test_y = self.y[train_index], self.y[test_index]
            clf.fit(train_X, train_y)
            predict_y = clf.predict(test_X)
            sn, sp, accuracy, mcc, rocAuc, k = self.calculateResult(test_y, predict_y)
            snList.append(sn)
            spList.append(sp)
            accuracyList.append(accuracy)
            mccList.append(mcc)
            rocAucList.append(rocAuc)
            kList.append(k)
        print("Sn: ", statistics.mean(snList))
        print("Sp: ", statistics.mean(spList))
        print("Accuracy: ", statistics.mean(accuracyList))
        print("MCC: ", statistics.mean(mccList))
        print("ROC-AUC: ", statistics.mean(rocAucList))
        print("K: ", statistics.mean(kList))
        return statistics.mean(snList), statistics.mean(spList), statistics.mean(accuracyList), statistics.mean(mccList), statistics.mean(rocAucList), statistics.mean(kList)

    def fitModel(self, clf, model):
        clf.fit(self.X, self.y)
        model = './model/' + model + '_' + str(self.trees) + '.pkl'
        with open(model, 'wb') as file:
            pickle.dump(clf, file)
            
    def calculateResult(self, test_y, predict_y):
        tn, fp, fn, tp = confusion_matrix(test_y, predict_y).ravel()
        sn = (tp / (tp + fn)) * 100
        sp = (tn / (fp + tn)) * 100
        accuracy = ((tp + tn) / (tp + fn + fp + tn)) * 100
        mcc = (((tp * tn) - (fp * fn)) / (math.sqrt((tp + fp)
                                                    * (tp + fn) * (fp + tn) * (tn + fn)))) * 100
        rocAuc = roc_auc_score(test_y, predict_y) * 100
        k = cohen_kappa_score(test_y, predict_y) * 100
        return sn, sp, accuracy, mcc, rocAuc, k

    def writeCSVHeader(self, method, fold, model):
        file = open("./report/MATLAB_" + model + "_" + method + "_" + str(fold) + ".csv", "a")
        file.write("ID,Sn,Sp,Accuracy,MCC,ROC-AUC,K,Details")
        file.write("\n")
        file.close()

    def writeCSVContent(self, sn, sp, accuracy, mcc, rocAuc, k, i, method, fold, model, trees):
        file = open("./report/MATLAB_" + model + "_" + method + "_" + str(fold) + ".csv", "a")
        details = method + " " + str(fold) + " Fold and " + str(trees) + " Trees."
        file.write(str(i) + "," + str(sn) + "," + str(sp) + "," + str(accuracy) + "," + str(mcc) + "," + str(rocAuc) + "," + str(k) + "," + details)
        file.write("\n")
        file.close()

    def modelSelector(self, model, trees):
        if (model == "RandomForestClassifier"):
            clf = RandomForestClassifier(
                n_estimators=trees, n_jobs=self.nJobs)
        elif (model == "BaggingClassifier"):
            clf = BaggingClassifier(n_estimators=trees, n_jobs=self.nJobs)
        elif (model == "ExtraTreesClassifier"):
            clf = ExtraTreesClassifier(
                n_estimators=trees, n_jobs=self.nJobs)
        elif (model == "RandomTreesEmbedding"):
            clf = RandomTreesEmbedding(
                n_estimators=trees, n_jobs=self.nJobs)
        elif (model == "AdaBoostClassifier"):
            clf = AdaBoostClassifier(n_estimators=trees)
        elif (model == "GradientBoostingClassifier"):
            clf = GradientBoostingClassifier(n_estimators=trees)
        else:
            clf = RandomForestClassifier(
                n_estimators=trees, n_jobs=self.nJobs)
        print(clf)
        return clf

    def cvSelector(self, method="ShuffleSplit", fold=10):
        if (method == "ShuffleSplit"):
            cv = ShuffleSplit(n_splits=fold)
        elif (method == "StratifiedKFold"):
            cv = StratifiedKFold(n_splits=fold)
        elif (method == "StratifiedShuffleSplit"):
            cv = StratifiedShuffleSplit(n_splits=fold)
        elif (method == "RepeatedStratifiedKFold"):
            cv = RepeatedStratifiedKFold(n_splits=fold)
        else:
            cv = ShuffleSplit(n_splits=fold)
        return cv
