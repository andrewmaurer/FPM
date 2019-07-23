import numpy as np
import matplotlib.pyplot as plt

def tpr( y_pred, y_actual, threshold):
        y_rounded = np.array([ pred > threshold for pred in y_pred ])
        num_true_positive = np.logical_and(y_rounded, y_actual).sum()
        num_total_positive = y_actual.sum()
        return num_true_positive / num_total_positive


def fpr( y_pred, y_actual, threshold):
        y_rounded = np.array([ pred > threshold for pred in y_pred ])
        num_false_positive = np.logical_and(y_rounded, np.logical_not(y_actual)).sum()
        num_total_positive = np.logical_not(y_actual).sum()
        return num_false_positive / num_total_positive


def compute_auc(y_pred, y_actual):
        thresholds = np.arange(0,1,0.01)
        fprs = [ fpr(y_pred, y_actual, threshold) for threshold in thresholds]
        tprs = [tpr(y_pred, y_actual, threshold) for threshold in thresholds]
        rectangle_bases = np.array([ np.abs(fprs[i+1] - fprs[i]) for i in range(len(fprs) - 1) ])
        rectangle_heights = tprs[:-1]
        print(rectangle_bases)
        print(rectangle_heights)
        rectangle_areas = rectangle_bases * rectangle_heights
        return rectangle_areas.sum()


def plot_auc(y_pred, y_actual, include_change=True):
        thresholds = np.arange(0,1,0.01)
        fprs = [ fpr(y_pred, y_actual, threshold) for threshold in thresholds]
        tprs = [tpr(y_pred, y_actual, threshold) for threshold in thresholds]
        plt.plot(fprs, tprs)  
        if include_change:
                plt.plot([0,1], [0,1], color='orange')      
        
        
y_actual = np.array([0] * 75 + [1] * 100)
y_pred = np.concatenate([
        np.random.uniform(0,   0.7, 75 ), 
        np.random.uniform(0.3, 1.0, 100 )
])

tpr( y_pred, y_actual, 0.5)
fpr( y_pred, y_actual, 0.5)

compute_auc(y_pred, y_actual)
plot_auc(y_pred, y_actual, include_change=True)