import os
import pickle
import gzip
import copy
import random, time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from datetime import datetime, date
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix


def flatten(t):
    return [item for sublist in t for item in sublist]


def format_IR(IR_num, split_type):
    """
    make a string, which can be used when storing trained neural network, results, log files, etc.
    """
    if isinstance(IR_num[0], list):  # Check if the first element is a list
        inner = ''.join(str(num) for num in IR_num[0])
        outer = '{:02d}'.format(IR_num[1])
        return f'T{split_type}_IR_{inner}_{outer}'
    else:
        return f'T{split_type}_IR_' + '_'.join('{:03d}'.format(num) for num in IR_num)

    

def make_folders(year_start, split_type, num_class, addition_str):
    """
    Create folder structure: {year}_{addition}/{t{split}_c{num}_*}
    """
    parent_folder = str(year_start)+"_"+ addition_str
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
        
    log_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_log")
    net_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_net")
    train_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_loss")
    figure_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_curve")
    result_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_result")

    try:
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
        if not os.path.exists(net_folder):
            os.mkdir(net_folder)
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
    except FileExistsError:
        pass

    save_folders = [net_folder, train_folder, figure_folder, result_folder]
    return save_folders, log_folder


######### Plots ###############
def calculate_plot_ROC(true_labels, nn_outputs, user_parameter, figure_name, save_figure_folder):
    """
    Plot ROC for binary classification.
    """
    num_class, IR_num, split_type, out_norm = user_parameter
    figure_path=os.path.join(save_figure_folder, figure_name)

    # -------- PATCH: safety reshape --------
    true_labels = np.asarray(true_labels).reshape(-1)
    nn_outputs  = np.asarray(nn_outputs).reshape(-1)

    fpr, tpr, _ = roc_curve(true_labels, nn_outputs)
    roc_auc = auc(fpr, tpr)
    auc_score_number = roc_auc_score(true_labels, nn_outputs)
    
    plt.figure()
    lw = 1.5
    plt.plot(fpr, tpr, color='blue', lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    os.makedirs(save_figure_folder, exist_ok=True)
    plt.savefig(figure_path,dpi=600)
    plt.show()
    plt.close()
    
    return auc_score_number
