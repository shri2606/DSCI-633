from collections import Counter
import numpy as np

class my_evaluation:
    def __init__(self, predictions, actuals, pred_proba=None):
        self.predictions = predictions
        self.actuals = actuals
        self.pred_proba = pred_proba
        self.classes_ = set(actuals)
        self.confusion_matrix = None

    def confusion(self):
        if self.confusion_matrix is not None:
            return self.confusion_matrix

        self.confusion_matrix = {}
        for label in self.classes_:
            self.confusion_matrix[label] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        for pred, actual in zip(self.predictions, self.actuals):
            if pred == actual:
                self.confusion_matrix[actual]["TP"] += 1
            else:
                self.confusion_matrix[pred]["FP"] += 1
                self.confusion_matrix[actual]["FN"] += 1

        return self.confusion_matrix

    def accuracy(self):
        if self.confusion_matrix is None:
            self.confusion()
        
        total_correct = sum(self.confusion_matrix[label]["TP"] for label in self.classes_)
        total_samples = len(self.actuals)
        self.acc = total_correct / total_samples
        return self.acc

    def precision(self, target=None, average="macro"):
        if self.confusion_matrix is None:
            self.confusion()
        
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp + fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            if average == "micro":
                prec = self.accuracy()
            else:
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    prec += prec_label * ratio
        return prec

    def recall(self, target=None, average="macro"):
        if self.confusion_matrix is None:
            self.confusion()
        
        if target is not None:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if tp + fn == 0:
                rec = 0
            else:
                rec = float(tp) / (tp + fn)
        else:
            if average == "micro":
                rec = self.accuracy()
            else:
                rec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    if tp + fn == 0:
                        rec_label = 0
                    else:
                        rec_label = float(tp) / (tp + fn)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    rec += rec_label * ratio
        return rec
    
    def f1(self, target=None, average="macro"):
        if target:
            prec = self.precision(target=target, average=average)
            rec = self.recall(target=target, average=average)
            if prec + rec == 0:
                f1_score = 0
            else:
                f1_score = 2.0 * prec * rec / (prec + rec)
        else:
            if average == "micro":
                prec = self.precision(target=None, average=average)
                rec = self.recall(target=None, average=average)
                if prec + rec == 0:
                    f1_score = 0
                else:
                    f1_score = 2.0 * prec * rec / (prec + rec)
            elif average == "macro":
                f1_score = 0
                for label in self.classes_:
                    f1_label = self.f1(target=label)
                    f1_score += f1_label
                if len(self.classes_) == 0:
                    f1_score = 0
                else:
                    f1_score /= len(self.classes_)
            elif average == "weighted":
                f1_score = 0
                n = len(self.actuals)
                for label in self.classes_:
                    f1_label = self.f1(target=label)
                    ratio = Counter(self.actuals)[label] / float(n)
                    f1_score += f1_label * ratio
            else:
                 raise Exception("Unknown type of average.")
        return f1_score
    


    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if type(self.pred_proba) == type(None):
            return None
        else:
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]
                tp = 0
                fp = 0
                fn = Counter(self.actuals)[target]
                tn = len(self.actuals) - fn
                tpr = 0
                fpr = 0
                auc_target = 0
                for i in order:
                    if self.actuals[i] == target:
                        tp += 1
                        fn -= 1
                        tpr = tp / (tp + fn)
                    else:
                        fp += 1
                        tn -= 1
                        pre_fpr = fpr
                        fpr = fp / (fp + tn)
                        auc_target += (fpr - pre_fpr) * tpr
            else:
                raise Exception("Unknown target class.")

            return auc_target
