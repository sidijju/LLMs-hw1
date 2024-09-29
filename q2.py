import openml
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import matplotlib.pyplot as plt

class Question2:
    def __init__(self):
        dataset = self.get_dataset()
        X, y = self.process_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

        self.adaboost_classifier = self.fit_adaboost_classifier(X_train, y_train)
        self.logistic_regression_classifier = self.fit_logistic_regression_classifier(X_train, y_train)

        self.plot_roc(X_test, y_test)
        self.plot_pr(X_test, y_test)
        self.calculate_stats(X_test, y_test)

    def get_dataset(self):
        dataset = openml.datasets.get_dataset(1464,
                                            download_data=False, 
                                            download_qualities=False, 
                                            download_features_meta_data=False)
        X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        return X

    def process_dataset(self, dataset):
        labels = dataset['Class']
        X = dataset.drop('Class', axis=1)
        y = labels == '2'
        y = y.astype(int)
        return X, y

    def fit_adaboost_classifier(self, X, y):
        adaboost_classifier = AdaBoostClassifier(algorithm="SAMME")
        adaboost_classifier.fit(X, y)
        return adaboost_classifier

    def fit_logistic_regression_classifier(self, X, y):
        logistic_regression_classifier = LogisticRegression()
        logistic_regression_classifier.fit(X, y)
        return logistic_regression_classifier

    def get_classifier_fpr_tpr(self, classifier, X, y):
        y_score = classifier.decision_function(X)
        fpr, tpr, _ = roc_curve(y, y_score)
        return fpr, tpr
    
    def get_classifier_precision_recall(self, classifier, X, y):
        y_score = classifier.decision_function(X)
        precision, recall, _ = precision_recall_curve(y, y_score)
        return precision, recall

    def plot_roc(self, X, y):
        fpr_adaboost, tpr_adaboost = self.get_classifier_fpr_tpr(self.adaboost_classifier, X, y)
        fpr_logistic_regression, tpr_logistic_regression = self.get_classifier_fpr_tpr(self.logistic_regression_classifier, X, y)

        plt.figure()
        plt.plot(fpr_adaboost, tpr_adaboost, color="darkorange", label="Adaboost")
        plt.plot(fpr_logistic_regression, tpr_logistic_regression, color="purple", label="Logistic Regression")
        plt.plot(1, 1, marker='o', color='black', label='all positive classifier')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.savefig('q2-part2-roc.png')

    def plot_pr(self, X, y):
        precision_adaboost, recall_adaboost = self.get_classifier_precision_recall(self.adaboost_classifier, X, y)
        precision_logistic_regression, recall_logistic_regression = self.get_classifier_precision_recall(self.logistic_regression_classifier, X, y)

        plt.figure()
        plt.plot(recall_adaboost, precision_adaboost, color="darkorange", label="Adaboost")
        plt.plot(recall_logistic_regression, precision_logistic_regression, color="purple", label="Logistic Regression")
        plt.plot(1, min(precision_logistic_regression), marker='o', color='black', label='all positive classifier')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision recall")
        plt.legend(loc="upper right")
        plt.savefig('q2-part2-pr.png')
    
    def calculate_pr_gains(self, y, y_score):
        fpr, tpr, _ = roc_curve(y, y_score)
        precision, _, _ = precision_recall_curve(y, y_score)

        epsilon = 1e-6

        P = np.sum(y == 1)
        N = np.sum(y == 0)
        TP = tpr * P + epsilon
        FP = fpr * N
        FN = P - TP

        pi = min(precision)
        one_minus_pi = 1 - pi
        
        precision_gain = 1 - (pi/one_minus_pi) * (FP/TP)
        recall_gain = 1 - (pi/one_minus_pi) * (FN/TP)

        precision_gain = np.clip(precision_gain, 0, 1)
        recall_gain = np.clip(recall_gain, 0, 1)

        return precision_gain, recall_gain
    
    def calculate_stats(self, X, y):
        y_score_ab = self.adaboost_classifier.decision_function(X)
        y_score_lr = self.logistic_regression_classifier.decision_function(X)

        roc_auc_ab = roc_auc_score(y, y_score_ab)
        roc_auc_lr = roc_auc_score(y, y_score_lr)

        precision_ab, recall_ab = self.get_classifier_precision_recall(self.adaboost_classifier, X, y)
        precision_lr, recall_lr = self.get_classifier_precision_recall(self.logistic_regression_classifier, X, y)

        pr_auc_lr = auc(recall_lr, precision_lr)
        pr_auc_ab = auc(recall_ab, precision_ab)

        precision_gain_ab, recall_gain_ab = self.calculate_pr_gains(y, y_score_ab)
        precision_gain_lr, recall_gain_lr = self.calculate_pr_gains(y, y_score_lr)

        prg_auc_lr = auc(recall_gain_lr, precision_gain_lr)
        prg_auc_ab = auc(recall_gain_ab, precision_gain_ab)

        print("Adaboost AUROC: " + str(roc_auc_ab))
        print("Logistic Regression AUROC: " + str(roc_auc_lr))
        print("Adaboost AUPR: " + str(pr_auc_ab))
        print("Logistic Regression AUPR: " + str(pr_auc_lr))
        print("Adaboost AUPRG: " + str(prg_auc_ab))
        print("Logistic Regression AUPRG: " + str(prg_auc_lr))
        
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    q2 = Question2()