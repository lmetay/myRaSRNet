# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
from sklearn.metrics import recall_score
eps=np.finfo(float).eps


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores, cls_iu, m_1 = cm2score(self.sum)
        scores.update(cls_iu)
        scores.update(m_1)
        return scores


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    # TN = hist[0][0]
    # FN = hist[1][0]
    FP = hist[0][1]
    # TP = hist[1][1]

    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    mean_acc = np.nanmean(tp / (sum_a1 + np.finfo(np.float32).eps))

    # FP rate 实际不变，而检测是变化的
    FPR = FP / (sum_a1[0] + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * tp / (sum_a1 + sum_a0 + np.finfo(np.float32).eps)
    # ---------------------------------------------------------------------- #
    # 2. Mean IoU
    # ---------------------------------------------------------------------- #
    iou = tp / (sum_a1 + sum_a0 - tp + np.finfo(np.float32).eps) #返回各个类别的Iou
    mean_iou = np.nanmean(iou)  # 求各类别IoU的平均

    cls_iu = dict(zip(range(n_class), iou))

    # ---------------------------------------------------------------------- #
    # 3. Kappa Coefficient (KC)
    # ---------------------------------------------------------------------- #
    N = hist.sum()
    PRE = ( (sum_a1[1])*(sum_a0[1]) / (N**2) ) + ( (sum_a1[0])*(sum_a0[0]) / (N**2) )
    KC = (acc - PRE) / (1-PRE + np.finfo(np.float32).eps)

    return {'Overall_Acc': acc,
            'Mean_IoU': mean_iou}, cls_iu, \
           {
        'precision': precision[1],
        'recall': recall[1],
        'F1': F1[1],
        'MeanAcc': mean_acc,
        'FPR': FPR,
        'KC': KC} 
        

class RunningMetrics(object):
    def __init__(self, num_classes):
        """
        Computes and stores the Metric values from Confusion Matrix
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param num_classes: <int> number of classes
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def __fast_hist(self, label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < self.num_classes) #
        hist = np.bincount(self.num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_gts, label_preds):
        """
        Compute Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gts: <np.ndarray> ground-truths
        :param label_preds: <np.ndarray> predictions
        :return:
        """
        for lt, lp in zip(label_gts, label_preds):
            self.confusion_matrix += self.__fast_hist(lt.flatten(), lp.flatten())

    def reset(self):
        """
        Reset Confusion Matrix
        :return:
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_cm(self):
        return self.confusion_matrix

    def get_scores(self):
        """
        Returns score about:
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :return:
        """
        hist = self.confusion_matrix
        n_class = hist.shape[0]
        # TN = hist[0][0]
        # FN = hist[1][0]
        FP = hist[0][1]
        # TP = hist[1][1]

        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)
        sum_a0 = hist.sum(axis=0)
        # ---------------------------------------------------------------------- #
        # 1. Accuracy & Class Accuracy
        # ---------------------------------------------------------------------- #
        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
        recall = tp / (sum_a1 + np.finfo(np.float32).eps)
        precision = tp / (sum_a0 + np.finfo(np.float32).eps)
        mean_acc = np.nanmean(tp / (sum_a1 + np.finfo(np.float32).eps))

        # FP rate 实际不变，而检测是变化的
        FPR = FP / (sum_a1[0] + np.finfo(np.float32).eps)

        # F1 score
        F1 = 2 * tp / (sum_a1 + sum_a0 + np.finfo(np.float32).eps)
        # ---------------------------------------------------------------------- #
        # 2. Mean IoU
        # ---------------------------------------------------------------------- #
        iou = tp / (sum_a1 + sum_a0 - tp + np.finfo(np.float32).eps) #返回各个类别的Iou
        mean_iou = np.nanmean(iou)  # 求各类别IoU的平均

        cls_iu = dict(zip(range(n_class), iou))

        # ---------------------------------------------------------------------- #
        # 3. Kappa Coefficient (KC)
        # ---------------------------------------------------------------------- #
        N = hist.sum()
        PRE = ( (sum_a1[1])*(sum_a0[1]) / (N**2) ) + ( (sum_a1[0])*(sum_a0[0]) / (N**2) )
        KC = (acc - PRE) / (1-PRE + np.finfo(np.float32).eps)

        return {'Overall_Acc': acc,
                'Mean_IoU': mean_iou}, cls_iu, \
                {
                'precision': precision[1],
                'recall': recall[1],
                'F1': F1[1],
                'MeanAcc': mean_acc,
                'FPR': FPR,
                'KC': KC} 

