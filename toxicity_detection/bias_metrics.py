# Unintended bias evaluation metrics as described in Dixon et al. 2018 (https://doi.org/10.1145/3278721.3278729) and Borkan et al. 2019 (https://doi.org/10.1145/3308560.3317593). Some of the implementations are inspired by the code at https://github.com/conversationai/unintended-ml-bias-analysis/blob/main/archive/unintended_ml_bias/model_bias_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from typing import Dict, List, Tuple

def compute_confusion_counts(y_actual:np.ndarray, y_pred:np.ndarray) -> Dict[str, int]:
    """Computes confusion counts, i.e. true positive, true negative, false positive, and false negative.

    Args:
        y_actual (np.ndarray): actual labels
        y_pred (np.ndarray): predicted labels

    Returns:
        Dict[str, int]: keys = metric name, values = counts
    """
    tp = sum([a==1 and p==1 for (a, p) in zip(y_actual, y_pred)])
    tn = sum([a==0 and p==0 for (a, p) in zip(y_actual, y_pred)])
    fp = sum([a==0 and p==1 for (a, p) in zip(y_actual, y_pred)])
    fn = sum([a==1 and p==0 for (a, p) in zip(y_actual, y_pred)])
    return {"tp":tp, "tn":tn, "fp":fp, "fn":fn}

def compute_confusion_rates(y_actual:np.ndarray, y_pred:np.ndarray) -> Dict[str, int]:
    """Compute the confusion rates, i.e. true positive rate, true negative rate, false positive rate, and false negative rate.

    Args:
        y_actual (np.ndarray): actual labels
        y_pred (np.ndarray): predicted labels

    Returns:
        Dict[str, int]: keys = metric name, values = rates
    """
    conf_counts = compute_confusion_counts(y_actual, y_pred)
    
    # true positive rate (= sensitivity, recall, hit rate)
    tpr = conf_counts["tp"] / (conf_counts["tp"] + conf_counts["fn"])

    # true negative rate (= specificity, selectivity)
    tnr = conf_counts["tn"] / (conf_counts["tn"] + conf_counts["fp"])
    
    # false positive rate (= fall-out)
    fpr = 1 - tnr
    
    # false negative rate (= miss rate)
    fnr = 1 - tpr
    
    return {
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr
    }

def compute_per_term_metric(grouping_cond:str, metric_name:str, data:pd.DataFrame, n_models:int=10, overall:bool=True, baseline:bool=False, model_name:str="orig") -> Dict[str, Dict[str, float]]:
    """Compute termwise scores of a specific metric for each of the n model variants. If overall=True, then the overall confusion rates for each model are also computed and returned. 

    Args:
        grouping_cond (str): the column to group the dataframe by.
        metric_name (str): the name of the metric to compute.
        data (pd.DataFrame): a pandas dataframe predictions made by each model.
        n_models (int, optional): the number of model variants. Defaults to 10.
        overall (bool, optional): whether to include overall scores or not. Defaults to True.
        baseline (bool, optional): is the model a single baseline or not? Defaults to False.
        model_name (str, optional): name of the model in the columns after "pred". Defaults to "orig".

    Returns:
        Dict[str, Dict[str, float]]: a nested dictionary of the format: {model_number: {term: score}}
    """
    # split data by identity term group
    term_groups = data.groupby(grouping_cond)
    
    # initialize dictionaries
    per_term_scores = dict()
    if overall:
        overall_scores = dict()
    
    if baseline:
        overall_scores["baseline"] = compute_confusion_rates(data["toxic"], data["baseline_pred"])[metric_name]
        t_scores = dict()
        for (t_name, t_df) in term_groups: # for each term (t = term)
            t_scores[t_name] = compute_confusion_rates(t_df["toxic"], t_df["baseline_pred"])[metric_name]
        per_term_scores["baseline"] = t_scores
    
    else:
        for i in range(n_models): # for each model variant
            
            # compute overall score for this model variant
            if overall: 
                overall_scores[str(i)] = compute_confusion_rates(data["toxic"], data[model_name+"_pred"+str(i)])[metric_name]
                
            # compute scpres for each term for this model variant
            t_scores = dict()
            for (t_name, t_df) in term_groups: # for each term (t = term)
                t_scores[t_name] = compute_confusion_rates(t_df["toxic"], t_df[model_name+"_pred"+str(i)])[metric_name]
            per_term_scores[str(i)] = t_scores
    
    if overall:
        return per_term_scores, overall_scores
    return per_term_scores

def plot_per_term_metrics(per_term_dict:Dict[str, Dict[str, float]], order:List[str]=None, title:str=None, save_name:str=None, grid:bool=False, n_models:int=10, FIGSIZE:Tuple[int]=(12,3)):
    """Plot scatterplot of per term metric scores.

    Args:
        per_term_dict (Dict[str, Dict[str, float]]): a nested dictionary containing the metric scores in the format: {model_number: {term: score}}.
        order (List[str], optional): a list of strings specifying the order of the terms on the x-axis. If none, they're sorted by their means (descending). Defaults to None.
        title (str, optional): the title of the plot. Defaults to None.
        save_name (str): the name to save the plot as. If None, the plot will not be saved. Defaults to None.
        grid (bool, optional): whether to add grid or not. Defaults to False.
        n_models (int, optional): the number of model variants. Defaults to 10.
        FIGSIZE (Tuple[int], optional): the figure size. Defaults to (12,3).
    """
    plt.figure(figsize=FIGSIZE)
    if order is None:
        order = list(pd.DataFrame(per_term_dict).mean(axis=1).sort_values(ascending=False).index)
    plot_df = pd.DataFrame(per_term_dict).T[order]
    for colname in plot_df:
        plt.scatter(x=[colname]*n_models, y=plot_df[colname], zorder=2, s=18)
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xticks(rotation=90, ha="center")
    if grid:
        plt.grid(zorder=1)
    plt.title(title)
    if save_name is not None:
        plt.savefig("plots\\"+save_name, bbox_inches="tight")
    plt.show()

def compute_fped(overall_fpr:float, per_term_fprs:Dict[str, float]) -> float:
    """Compute the False Positive Equality Difference.

    Args:
        overall_fpr (float): false positive rate (fpr) for entire dataset
        per_term_fprs (Dict[str, float]): a dictionary with the format: {identity term: fpr for subset of data that contains that term}

    Returns:
        float: the FPED value
    """
    fped = sum( abs(overall_fpr - fpr_t) for fpr_t in per_term_fprs.values() )
    return fped

def compute_fned(overall_fnr:float, per_term_fnrs:Dict[str, float]) -> float:
    """Compute the False Negative Equality Difference.

    Args:
        overall_fnr (float): false negative rate (fnr) for entire dataset
        per_term_fnrs (Dict[str, float]): a dictionary with the format: {identity term: fpr for subset of data that contains that term}

    Returns:
        float: the FNED value
    """
    fned = sum( abs(overall_fnr - fnr_t) for fnr_t in per_term_fnrs.values() )
    return fned

def normalized_mwu(data_A:pd.DataFrame, data_B:pd.DataFrame, col_name:str) -> float:
    """Calculate the normalized Mann-Whitney U test statistic for data_A and data_B.

    Args:
        data_A (pd.DataFrame): the first group of data.
        data_B (pd.DataFrame): the second group of data.
        col_name (int): the name of the column where the scores are located.

    Returns:
        float: the normalized Mann-Whitney U test statistic.
    """
    scores_A = data_A[col_name]
    scores_B = data_B[col_name]
    n_A = len(scores_A)
    n_B = len(scores_B)
    if n_A == 0 or n_B == 0:
        raise ZeroDivisionError("The length of data_A or data_B (or both) is zero.")
    
    u, p = mannwhitneyu(scores_A, scores_B, alternative="less") # u = statistic, p = p-value
    norm_mwu = u / (n_A * n_B)
    return norm_mwu

def compute_posAEG(subgroup_pos:pd.DataFrame, bgr_pos:pd.DataFrame, col_name:str) -> float:
    """Compute the positive average equality gap. 

    Args:
        subgroup_pos (pd.DataFrame): the positive examples in the subgroup data.
        bgr_pos (pd.DataFrame): the positive examples in the background data.
        col_name (str): the name of the column where the scores are located.

    Returns:
        float: the positive AEG score [-0.5, 0.5].
    """
    norm_mwu = normalized_mwu(bgr_pos, subgroup_pos, col_name) 
    if norm_mwu is None:
        return None
    pos_aeg = 0.5 - norm_mwu
    return pos_aeg
    
def compute_negAEG(subgroup_neg:pd.DataFrame, bgr_neg:pd.DataFrame, col_name:str) -> float:
    """Compute the negative average equality gap. 

    Args:
        subgroup_neg (pd.DataFrame): the negative examples in the subgroup data.
        bgr_neg (pd.DataFrame): the negative examples in the background data.
        col_name (str): the name of the column where the scores are located.

    Returns:
        float: the negative AEG score [-0.5, 0.5].
    """
    norm_mwu = normalized_mwu(bgr_neg, subgroup_neg, col_name) 
    if norm_mwu is None:
        return None
    neg_aeg = 0.5 - norm_mwu
    return neg_aeg

def power_mean(series:pd.Series, p:float) -> float:
    """Compute the power mean (generalized mean) of a series. 
    Adapted from: https://www.kaggle.com/code/dborkan/benchmark-kernel/notebook#Create-a-text-tokenizer

    Args:
        series (pd.Series): the series to compute the power mean for.
        p (float): the power value.

    Returns:
        float: the computed power mean.
    """
    total = sum(np.power(series, p))
    power_mean = np.power(total / len(series), 1 / p)
    return power_mean

def get_weighted_bias_score(bias_df:pd.DataFrame, overall_auc:float, POWER:float=-5, OVERALL_MODEL_WEIGHT:float=0.25, SUBMETRIC_WEIGHTS:List[float]=[0.25,0.25,0.25], SUBMETRIC_NAMES:List[str]=["SubAUC", "BPSN", "BNSP"]) -> float:
    """Calculate the weighted bias score by combining the overall AUC and the generalized mean of the bias submetrics. See description at https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation 
    Adapted from: https://www.kaggle.com/code/dborkan/benchmark-kernel/notebook#Create-a-text-tokenizer 

    Args:
        bias_df (pd.DataFrame): DataFrame containing the per-term bias submetric scores, e.g. subAUC, BPSN, and BNSP.
        overall_auc (float): ROC-AUC on the entire synthetic dataset. 
        POWER (float, optional): the power value. Defaults to -5.
        OVERALL_MODEL_WEIGHT (float, optional): the weight given to the overall model AUC. Defaults to 0.25.
        SUBMETRIC_WEIGHTS (List, optional): the weights given to each of the submetrics. Defaults to [0.25,0.25,0.25].
        SUBMETRIC_NAMES (List, optional): a list of the submetric names as they are used in the bias_df. Defaults to ["subAUC_avg", "BPSN_avg", "BNSP_avg"].

    Returns:
        float: the weighted bias score.
    """
    bias_score = OVERALL_MODEL_WEIGHT * overall_auc + np.sum([
        SUBMETRIC_WEIGHTS[0]*power_mean(bias_df[SUBMETRIC_NAMES[0]], POWER), # e.g. subAUC
        SUBMETRIC_WEIGHTS[1]*power_mean(bias_df[SUBMETRIC_NAMES[1]], POWER), # e.g. BPSN
        SUBMETRIC_WEIGHTS[2]*power_mean(bias_df[SUBMETRIC_NAMES[2]], POWER)  # e.g. BNSP
    ])
    return bias_score