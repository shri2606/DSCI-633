import numpy as np
import pandas as pd

class my_evaluation:
    def __init__(self, predictions, actuals, pred_proba=None):
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        self.classes_ = sorted(list(set(self.predictions) | set(self.actuals)))

    def _calc_tp_fp_fn(self, target_class):
        tp = np.sum((self.predictions == target_class) & (self.actuals == target_class))
        fp = np.sum((self.predictions == target_class) & (self.actuals != target_class))
        fn = np.sum((self.predictions != target_class) & (self.actuals == target_class))
        return tp, fp, fn

    def precision(self, target=None, average="macro"):
        def calc_precision(tp, fp):
            return tp / (tp + fp) if (tp + fp) > 0 else 0

        if target:
            tp, fp, _ = self._calc_tp_fp_fn(target)
            return calc_precision(tp, fp)
        else:
            precisions = []
            for cls in self.classes_:
                tp, fp, _ = self._calc_tp_fp_fn(cls)
                precisions.append(calc_precision(tp, fp))
            return np.mean(precisions) if average == "macro" else np.average(precisions, weights=[self._calc_tp_fp_fn(cls)[0] + self._calc_tp_fp_fn(cls)[2] for cls in self.classes_])

    def recall(self, target=None, average="macro"):
        def calc_recall(tp, fn):
            return tp / (tp + fn) if (tp + fn) > 0 else 0

        if target:
            tp, _, fn = self._calc_tp_fp_fn(target)
            return calc_recall(tp, fn)
        else:
            recalls = []
            for cls in self.classes_:
                tp, _, fn = self._calc_tp_fp_fn(cls)
                recalls.append(calc_recall(tp, fn))
            return np.mean(recalls) if average == "macro" else np.average(recalls, weights=[self._calc_tp_fp_fn(cls)[0] + self._calc_tp_fp_fn(cls)[2] for cls in self.classes_])

    def f1(self, target=None, average="macro"):
        def calc_f1(precision, recall):
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if target:
            precision = self.precision(target)
            recall = self.recall(target)
            return calc_f1(precision, recall)
        else:
            f1_scores = []
            for cls in self.classes_:
                precision = self.precision(cls)
                recall = self.recall(cls)
                f1_scores.append(calc_f1(precision, recall))
            return np.mean(f1_scores) if average == "macro" else np.average(f1_scores, weights=[self._calc_tp_fp_fn(cls)[0] + self._calc_tp_fp_fn(cls)[2] for cls in self.classes_])

    # ... [AUC implementation] ...

    def auc(self, target):
        if self.pred_proba is None or target not in self.classes_:
            return None

        
        actuals_target = (self.actuals == target).astype(int)
        probas_target = self.pred_proba[target]
        sorted_indices = np.argsort(probas_target)[::-1]
        sorted_actuals = actuals_target[sorted_indices]
        
        
        tpr = []
        fpr = []
        positive_count = np.sum(actuals_target)
        negative_count = len(actuals_target) - positive_count
        tp = 0
        fp = 0
        for i in range(len(sorted_actuals)):
            if sorted_actuals[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr.append(tp / positive_count if positive_count else 0)
            fpr.append(fp / negative_count if negative_count else 0)

        
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
        return auc


