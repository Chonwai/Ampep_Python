import sklearn
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, RandomTreesEmbedding, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.model_selection import KFold, GroupKFold
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

    def training(self, fold=10, trees=100, model="RandomForestClassifier", method="KFold"):
        self.trees = trees
        self.writeCSVHeader(method, fold, model)
        for i in range(1, 31):
            trees = self.trees * i
            clf = self.modelSelector(model, trees)
            cv = self.cvSelector(method, fold)
            # score = cross_val_score(clf, self.X, self.y, cv=cv, n_jobs=4, scoring='accuracy')
            score = cross_val_predict(clf, self.X, self.y, cv=cv, n_jobs=4)
            # score = cross_validate(clf, self.X, self.y, cv=cv, n_jobs=4, return_train_score=True)
            clf.fit(self.X, self.y)
            print("Finished Training Model " + str(i) + " Times with " +
                  str(fold) + " Fold and " + str(trees) + " Trees!")
            # model = './model/Matlab/' + method + '/RF_' + method + '_' + str(fold) + '_' + str(trees + 100 * i) + '.pkl'
            # with open(model, 'wb') as file:
            #     pickle.dump(clf, file)
            tn, fp, fn, tp = confusion_matrix(self.y, score).ravel()

            sn = (tp / (tp + fn)) * 100
            sp = (tn / (fp + tn)) * 100
            accuracy = ((tp + tn) / (tp + fn + fp + tn)) * 100
            mcc = (((tp * tn) - (fp * fn)) / (math.sqrt((tp + fp)
                                                        * (tp + fn) * (fp + tn) * (tn + fn)))) * 100
            rocAuc = roc_auc_score(self.y, score) * 100
            k = cohen_kappa_score(self.y, score) * 100

            print("Sn: ", sn)
            print("Sp: ", sp)
            print("Accuracy: ", accuracy)
            print("MCC: ", mcc)
            print("ROC-AUC: ", rocAuc)
            print("K: ", k)

            self.writeCSVContent(sn, sp, accuracy, mcc, rocAuc, k, i, method, fold, model, trees)
            # self.calculateScore(score, fold, scoring=['accuracy'])

    def writeCSVHeader(self, method, fold, model):
        file = open("./report/" + model + "_" + method + "_" + str(fold) + ".csv", "a")
        file.write("ID,Sn,Sp,Accuracy,MCC,ROC-AUC,K,Details")
        file.write("\n")
        file.close()

    def writeCSVContent(self, sn, sp, accuracy, mcc, rocAuc, k, i, method, fold, model, trees):
        file = open("./report/" + model + "_" + method + "_" + str(fold) + ".csv", "a")
        details = method + " " + str(fold) + " Fold and " + str(trees) + " Trees."
        file.write(str(i) + "," + str(sn) + "," + str(sp) + "," + str(accuracy) + "," + str(mcc) + "," + str(rocAuc) + "," + str(k) + "," + details)
        file.write("\n")
        file.close()

    def calculateScore(self, score, fold, scoring):
        name = 'test_score'
        print(name)
        meanAcc = statistics.mean(score)
        print("Mean Accuracy: " + str(meanAcc))
        print("Top Accuracy: " + str(max(score) * 100))
        print("Bottom Accuracy: " + str(min(score) * 100))
        print("")

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

    def cvSelector(self, method="KFold", fold=10):
        if (method == "KFold"):
            cv = KFold(n_splits=fold)
        elif (method == "GroupKFold"):
            cv = GroupKFold(n_splits=fold)
        elif (method == "ShuffleSplit"):
            cv = ShuffleSplit(n_splits=fold)
        elif (method == "StratifiedKFold"):
            cv = StratifiedKFold(n_splits=fold)
        elif (method == "StratifiedShuffleSplit"):
            cv = StratifiedShuffleSplit(n_splits=fold)
        elif (method == "RepeatedStratifiedKFold"):
            cv = RepeatedStratifiedKFold(n_splits=fold)
        else:
            cv = KFold(n_splits=fold)
        return cv
