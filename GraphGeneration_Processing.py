import math
import random
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
    

from numpy import random
from scipy import stats as st

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 8]

import logging
# logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import random

def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

def time_part(timepartsize, nodeslistsize):
    time_partition = constrained_sum_sample_pos(timepartsize,nodeslistsize)
    return time_partition


def create_input_file(num_nodes, desired_mean_edges, max_nodes, timepartsize, descr_string):
    
    txt_file_path = "ListSize" + str(max_nodes) \
    + "MeanEdges" \
    +  str(desired_mean_edges) + str(descr_string)
    
    # create simple list of nodes - numbers from 1 -> MaxNodes
    nodes_list = list(range(1,num_nodes))
    nodeslistsize = max_nodes
    random.seed(123)
    desired_mean_edges = desired_mean_edges #low noise #
    freq_edges = np.random.geometric(1/desired_mean_edges, size=nodeslistsize)
    logger.info("Edge frequencies: {}".format(freq_edges))
    logger.info("Stats Summary of freq_edges: {}".format(st.describe(freq_edges)))
    
    # create time partition
    time_part1 = time_part(timepartsize,nodeslistsize)
    logger.info("Time Partion created of length {}".format(len(time_part1)))
    logger.info("Time Partion List glimpse: {}".format(time_part1))

    # initialize empty dataframe
    df_freq_count = pd.DataFrame(columns = ["timestamp", "src", "dest", "weight", "label", "freq"])
    logger.info("Empty Dataframe initialized for output")

    # repeat out time_partition to get timestamp from index
    time_part1 = pd.Series(time_part1)
    time_part1 = time_part1.index.repeat(time_part1)
    logger.info("Time Partition Dataframe Head: {}".format)
    time_part1 = time_part1.tolist()
    
    # count of uniques
    freq_edges = pd.Series(freq_edges)
    

    # for each item in freq_edges:
    for index, val in freq_edges.items():

        # put two numbers from nodes list as src, dest
        temp_src, temp_dest = random.sample(nodes_list,2)

        # set time stamp from INDEX of freq_edges
        temp_time = time_part1[index]
    #     print("temp time: ", temp_time)
        
        # label = anomaly rule
        temp_label = 1 if val > (desired_mean_edges*2) else 0
        
    # add this stuff to df
        df_freq_count = df_freq_count.append({"timestamp":temp_time, 
                              "src": temp_src,
                              "dest": temp_dest,
                              "weight": 1,
                              "label": temp_label,
                              "freq": val}
                             , ignore_index=True )
    
    # save pic of freq_count with descr_string
    ax = plt.gca()
    current_fig = df_freq_count.groupby('timestamp').sum().plot(
    kind='line', 
    y='freq',
    ax=ax,
    legend=False)
    ax.set_xlabel('Timestamp', fontsize=18)
    ax.set_ylabel('Total Edge Frequency', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig = current_fig.get_figure()
    fig.savefig(txt_file_path+".png")
    logger.info("See image: ".format(txt_file_path))
    
    # repeat freq count rows out 
    df_freq_count['freq']=df_freq_count['freq'].astype(np.int64)
    df_input = df_freq_count.loc[df_freq_count.index.repeat(df_freq_count['freq'])]
    df_input=df_input.reset_index(drop=True)
    df_input = df_input[['timestamp','src', 'dest','weight', 'label']]

    df_input.to_csv(txt_file_path + ".csv", index = False, sep=",", header=False)
    logger.info("Done...CSV written to: {}".format(txt_file_path))
    return df_input, df_freq_count
    
def import_test_data(txt_file_path_str, sep_str):
    df = pd.read_csv(txt_file_path_str, 
    sep=sep_str, header=None, names=["timestamp", "src", "dest", "weight", "label"],
    index_col = False)
    df = df.drop(['weight'], axis=1)
    df = df.sort_values(by=["timestamp", "src", "dest"])
    df = df.groupby(["timestamp", "src", "dest"]).max()
    return df

def import_output_data(txt_file_path_str, sep_str):
    df = pd.read_csv(txt_file_path_str, sep=sep_str, 
        header=None, names=["timestamp", "src", "dest", "score"])
    df = df.sort_values(by=["timestamp", "src", "dest"])
    df = df.groupby(["timestamp", "src", "dest"]).max()
    return df

def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])
    
    
# Here's the custom function returning classification report dataframe:
def metrics_report_to_df(ytrue, ypred, dataset_name):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(ytrue, ypred)
    accuracy = metrics.accuracy_score(ytrue, ypred)
    classification_report = pd.concat(map(pd.DataFrame, [precision, 
    recall, 
    fscore, 
    support]), axis=1)
    classification_report.columns = ["precision", "recall", "f1-score", "support"] # Add row w "avg/total"
    classification_report.loc['avg/Total', :] = metrics.precision_recall_fscore_support(ytrue, 
    ypred, average='weighted')
    classification_report.loc['avg/Total', 'support'] = classification_report['support'].sum() 
    classification_report['Name']= dataset_name
    classification_report['accuracy'] = np.nan
    classification_report.loc['avg/Total', 'accuracy'] = accuracy 
    return(classification_report)
    
def process_accuracy_scores(test_file_path, output_file_path, threshold=0.8):
    # initialize dataframe for data output-> time,
    df_test = import_test_data(test_file_path, sep_str = ",")
    dataset_name = str(test_file_path)
    df_out = import_output_data(output_file_path, sep_str = ",")

    # check length of files
    if len(df_test) == len(df_out):
        print("Test_Output lengths match...keep going...")
    else:
        raise Exception("Y-Test vs Y-Pred length mismatch")

    # assume 'label' column indicates 1 = anomaly, 0 = not - CHECK this
    print("Checking label column exists & is binary...")
    if np.isin(df_test["label"].dropna().unique(),[0, 1]).all():
          print("Label format is binary - proceed...")
          y_true = df_test['label']
    else:
          raise Exception("Label format either doesn't exist or is not binary - please fix!")

    # create column for preds based on threshold
    y_pred = (df_out['score'] >= threshold)
    print(y_true, y_pred)
    
    # Provide input as true_label and predicted label (from process_accuracy_scores)
    classification_report = metrics_report_to_df(y_true, y_pred, dataset_name)
    
    # save confusion matrix in sep dataframe
    try:
        TP, FP, FN, TN = get_confusion_matrix_values(y_true, y_pred)
    except Exception:
            TP, FP, FN, TN = np.nan, np.nan, np.nan, np.nan
    
    # initialize df for confusion matrix
    cm_asdf = pd.DataFrame(data = {'TP': TP,
                                   'FP': FP,
                                   'FN': FN ,
                                   'TN': FN,
                                   'Dataset': dataset_name
                                  }, index=[0])
    
    # return both
    return cm_asdf, classification_report
 
 def process_enron(enron_test, enron_out, threshold=0.8):
    # initialize dataframe for data output-> time,
    df_test = enron_test
    dataset_name = "Enron"
    df_out = enron_out

    # check length of files
    if len(df_test) == len(df_out):
        print("Test_Output lengths match...keep going...")
    else:
        raise Exception("Y-Test vs Y-Pred length mismatch")

    # assume 'label' column indicates 1 = anomaly, 0 = not - CHECK this
    print("Checking label column exists & is binary...")
    if np.isin(df_test["label"].dropna().unique(),[0, 1]).all():
          print("Label format is binary - proceed...")
          y_true = df_test['label']
    else:
          raise Exception("Label format either doesn't exist or is not binary - please fix!")

    # create column for preds based on threshold
    y_pred = (df_out['score'] >= threshold)
    print(y_true, y_pred)
    
    # Provide input as true_label and predicted label (from process_accuracy_scores)
    classification_report = metrics_report_to_df(y_true, y_pred, dataset_name)
    
    # save confusion matrix in sep dataframe
    TP, FP, FN, TN = get_confusion_matrix_values(y_true, y_pred)
    
    # initialize df for confusion matrix
    cm_asdf = pd.DataFrame(data = {'TP': TP,
                                   'FP': FP,
                                   'FN': FN ,
                                   'TN': FN,
                                   'Dataset': dataset_name
                                  }, index=[0])
    
    # return both
    return cm_asdf, classification_report
    
