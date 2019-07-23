import numpy as np
import matplotlib.pyplot as plt

def tpr( y_pred, y_actual, threshold):
        """
        y_pred    : numpy array of predicted values between 0 and 1.
        y_actual  : numpy array of actual values, either 0 or 1.
        threshold : value at which roundup begins.

        Returns true positive rate when `y_pred` is rounded up when greater than `threshold` and down otherwise.
        """
        y_rounded = np.array([ pred > threshold for pred in y_pred ])
        num_true_positive = np.logical_and(y_rounded, y_actual).sum()
        num_total_positive = y_actual.sum()
        return num_true_positive / num_total_positive


def fpr( y_pred, y_actual, threshold):
        """
        y_pred    : numpy array of predicted values between 0 and 1.
        y_actual  : numpy array of actual values, either 0 or 1.
        threshold : value at which roundup begins.

        Returns false positive rate when `y_pred` is rounded up when greater than `threshold` and down otherwise.
        """
        y_rounded = np.array([ pred > threshold for pred in y_pred ])
        num_false_positive = np.logical_and(y_rounded, np.logical_not(y_actual)).sum()
        num_total_positive = np.logical_not(y_actual).sum()
        return num_false_positive / num_total_positive


def compute_auc(y_pred, y_actual):
        """
        Approximates the Area Under the Curve of the Receiver Operating Characteristic.

        roc_auc_score from sklearn.metrics is definitely a closer approximation. Will improve here.
        """
        thresholds = np.arange(0,1,0.01)
        fprs = [ fpr(y_pred, y_actual, threshold) for threshold in thresholds]
        tprs = [tpr(y_pred, y_actual, threshold) for threshold in thresholds]
        rectangle_bases = np.array([ np.abs(fprs[i+1] - fprs[i]) for i in range(len(fprs) - 1) ])
        rectangle_heights = tprs[:-1]
        rectangle_areas = rectangle_bases * rectangle_heights
        return rectangle_areas.sum()


def plot_auc(y_pred, y_actual, include_chance=True):
        """
        include_chance : if True, plots the straight line y=x which can be compared to ROC score.

        Uses matplotlib.pyplot to plot the ROC curve.
        """
        thresholds = np.arange(0,1,0.01)
        fprs = [ fpr(y_pred, y_actual, threshold) for threshold in thresholds]
        tprs = [tpr(y_pred, y_actual, threshold) for threshold in thresholds]
        plt.plot(fprs, tprs)
        if include_chance:
                plt.plot([0,1], [0,1], color='orange')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        
        
# y_actual = np.array([0] * 75 + [1] * 100)
# y_pred = np.concatenate([
#         np.random.uniform(0,   0.7, 75 ), 
#         np.random.uniform(0.3, 1.0, 100 )
# ])

# tpr( y_pred, y_actual, 0.5)
# fpr( y_pred, y_actual, 0.5)

# compute_auc(y_pred, y_actual)
# plot_auc(y_pred, y_actual, include_chance=True)