# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:06:04 2021

@author: malvika
Pre-process the SNV data
"""


# Import 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


def getCivicLabels(data, DATAPATH):
    """
    Get CIViC labels along with neutral gene labels

    Parameters
    ----------
    data : DataFrame
        Must contain columns "Hugo_Symbol", "Chr", "Start", "End", "Ref", and
        "Alt".
    DATAPATH : str
        Path to PIVOT data folder.

    Returns
    -------
    data : DataFrame
        Contains a column "Label" that includes CIViC labels.

    """
    data = data.copy()
    # Load neutral genes
    os.chdir(DATAPATH)
    fname = "neutral.txt"
    neutral = list(pd.read_csv(fname, header=0).Neutral)

    # load civic filtered
    os.chdir(DATAPATH + "/driver genes/CIViC")
    fname = "civic_filtered.txt"
    data_civic = pd.read_csv(fname, sep="\t", header=0)
    
    #
    data.loc[:, "Label"] = ["Unlabelled"] * data.shape[0]
    data.loc[data.Hugo_Symbol.isin(neutral), "Label"]="Neutral"
    temp = ["Driver" if "{}:{}-{}|{}|{}".format(chrm, start, end, ref, mut) in 
            list(data_civic.Mut_loc) else lab for chrm, start, end, ref, mut, lab in 
            zip(data.Chr, data.Start, data.End, data.Ref,
                data.Alt, data.Label)]
    data.loc[:, "Label"] = temp
    return data


def getCGCLabels(data, DATAPATH):
    """
    Get CGC labels. 

    Parameters
    ----------
    data : DataFrame
        Must contain columns "Hugo_Symbol".
    DATAPATH : str
        Path to PIVOT data folder.

    Returns
    -------
    data : DataFrame
        Contains a column "Label" that includes CGC labels.

    """
    data = data.copy()

    # Load neutral genes
    os.chdir(DATAPATH)
    fname = "neutral.txt"
    neutral = list(pd.read_csv(fname, header=0).Neutral)

    # Load CGC data
    os.chdir(DATAPATH + "/driver genes/CGC/")
    fname = "cancer_gene_census_9nov2021.csv"
    data_cgc = pd.read_csv(fname, sep=",", header=0)

    # 
    data.loc[:, "Label"] = ["Unlabelled"] * data.shape[0]
    data.loc[data.Hugo_Symbol.isin(neutral), "Label"]="Neutral"
    tsg_list = list(data_cgc[data_cgc["Role in Cancer"] == "TSG"]["Gene Symbol"])
    og_list = list(data_cgc[data_cgc["Role in Cancer"] == "oncogene"]["Gene Symbol"])
    temp = ["Tumor suppressor" if gene in tsg_list else lab for gene, lab in zip(data.Hugo_Symbol, data.Label)]
    data.loc[:, "Label"] = temp
    temp = ["Oncogene" if gene in og_list else lab for gene, lab in zip(data.Hugo_Symbol, data.Label)]
    data.loc[:, "Label"] = temp
    return data


def getBaileyLabels(data, DATAPATH, ctype):
    """
    Get driver gene labels from Bailey et al.

    Parameters
    ----------
    data : DataFrame
        Must contain columns "Hugo_Symbol".
    DATAPATH : str
        Path to PIVOT data folder.
    ctype : str
        TCGA cancer type.

    Returns
    -------
    data : DataFrame
        Contains a column "Label" that includes Bailey et al. gene labels.

    """
    data = data.copy()

    # Load neutral genes
    os.chdir(DATAPATH)
    fname = "neutral.txt"
    neutral = list(pd.read_csv(fname, header=0).Neutral)
    
    # Load bailey data
    os.chdir(DATAPATH + "/driver genes/Bailey et al/")
    fname = "bailey_genes.txt"
    data_bailey = pd.read_csv(fname, sep="\t", header=0)
    if ctype == "COAD":
        data_bailey = data_bailey[data_bailey.Cancer == "COADREAD"]
    else:
        data_bailey = data_bailey[data_bailey.Cancer == ctype]


    #
    data.loc[:, "Label"] = ["Unlabelled"] * data.shape[0]
    data.loc[data.Hugo_Symbol.isin(neutral), "Label"]="Neutral"
    tsg_list = list(data_bailey[(data_bailey["Tumor suppressor or oncogene prediction (by 20/20+)"] == "tsg") | (data_bailey["Tumor suppressor or oncogene prediction (by 20/20+)"] == "possible tsg")]["Gene"])
    og_list = list(data_bailey[(data_bailey["Tumor suppressor or oncogene prediction (by 20/20+)"] == "oncogene") | (data_bailey["Tumor suppressor or oncogene prediction (by 20/20+)"] == "possible oncogene")]["Gene"])
    temp = ["Tumor suppressor" if gene in tsg_list else lab for gene, lab in zip(data.loc[:, "Hugo_Symbol"], data.loc[:, "Label"])]
    data.loc[:, "Label"] = temp
    temp = ["Oncogene" if gene in og_list else lab for gene, lab in zip(data.loc[:, "Hugo_Symbol"], data.loc[:, "Label"])]
    data.loc[:, "Label"] = temp
    return data


def getMartelottoLabels(data, DATAPATH):
    """
    Get labels from Martelotto gene labels

    Parameters
    ----------
    data : DataFrame
        Must contain columns "Hugo_Symbol", "Chr", "Start", "End", "Ref", and
        "Alt".
    DATAPATH : str
        Path to PIVOT data folder.

    Returns
    -------
    data : DataFrame
        Contains a column "Label" that includes Martelotto et al. gene labels.

    """
    data = data.copy()
    # Load neutral genes
    os.chdir(DATAPATH)
    fname = "neutral.txt"
    neutral = list(pd.read_csv(fname, header=0).Neutral)

    # Load martelotto data
    os.chdir(DATAPATH + "/driver genes/Martelotto et al/")
    fname = "martelotto_final.txt"
    data_martelotto = pd.read_csv(fname, sep="\t", header=0)

    #
    data.loc[:, "Label"] = ["Unlabelled"] * data.shape[0]
    data.loc[data.loc[:, "Hugo_Symbol"].isin(neutral), "Label"]="Neutral"
    # make dictionary for label
    label_mar = pd.DataFrame(index=data_martelotto.loc[:, "Mut_loc"], columns=["Label"]) 
    label_mar.loc[:, "Label"] = list(data_martelotto.loc[:,"Type"])
    temp = [label_mar.loc["{}:{}-{}|{}|{}".format(chrm, start, end, ref, mut),
                          "Label"] if "{}:{}-{}|{}|{}".format(chrm,
                            start, end, ref, mut) in 
            list(label_mar.index) else lab for chrm, start, end, ref, mut, lab in 
            zip(data.loc[:, "Chr"], data.loc[:, "Start"],
                data.loc[:, "End"], data.loc[:, "Ref"],
                data.loc[:, "Alt"], data.loc[:, "Label"])]
    data.loc[:, "Label"] = temp
    return data    


def getSNVFeatures(data, ctype, path_domains):
    """
    Converts mutation matirx into list of features.

    Parameters
    ----------
    data : Dataframe
        Data containing mutation information.
    ctype : str
        Cancer type.
    path_domains : str
        Path to driver domain file and pfam mapping of domains.

    Returns
    -------
    data : Dataframe
        Data containing mutation information.

    """
    data = data.copy()
    # Generate ordinal features
    cols_oe = ['SIFT_pred', 'Polyphen2_HDIV_pred',
               'Polyphen2_HVAR_pred', 'LRT_pred',
               'MutationTaster_pred', 'MutationAssessor_pred',
            'FATHMM_pred', 'PROVEAN_pred', 'fathmm-MKL_coding_pred',
            'MetaSVM_pred', 'MetaLR_pred']
    data[cols_oe] = data[cols_oe].fillna('-') 
    oenc = OrdinalEncoder(categories=[['-', 'T', 'D'], ['-', 'B', 'P', 'D'],
                                     ['-', 'B', 'P', 'D'], ['-', 'N', 'U', 'D'],
                                     ['-', 'P', 'N', 'A', 'D'], ['-', 'N', 'L', 'M', 'H'],
                                     ['-', 'T', 'D'], ['-', 'N', 'D'],  ['-', 'N', 'D'],
                                     ['-', 'T', 'D'], ['-', 'T', 'D']])
    oenc.fit(data[cols_oe])
    data[cols_oe] = oenc.transform(data[cols_oe])
    
    # Genrate one hot encoding features
    cols_ohe = ['Variant_Classification']
    ohenc = OneHotEncoder(categories=[['Missense_Mutation', 'Silent',
                                      'Nonsense_Mutation', 'Frame_Shift_Del',
                                      'Intron', 'RNA', 'Frame_Shift_Ins',
                                      "5'UTR", "3'UTR", "3'Flank",
                                      'Splice_Site', 'Splice_Region', "5'Flank",
                                      'In_Frame_Del', 'Translation_Start_Site',
                                      'Nonstop_Mutation', 'In_Frame_Ins',
                                      'IGR']],
                          sparse=False,
                          handle_unknown='ignore')
    ohenc.fit(data[cols_ohe])
    cols_op_ohe = ["{}_{}".format(x, y) for i, x in enumerate(cols_ohe) for y in list(ohenc.categories_[i])]
    for col in cols_op_ohe:
        data[col] = [None] * data.shape[0]
    data[cols_op_ohe] = ohenc.transform(data[cols_ohe])

    
    # Add domain features
    os.chdir(path_domains)
    fname="ct_pfamdomains.txt"
    ct_pfam = pd.read_csv(fname, sep="\t", header=0, index_col=0)
    os.chdir(path_domains)
    fname="Pfam-A.clans.tsv"
    pfam_map = pd.read_csv(fname, sep="\t", header=None, index_col=3)
    ## Get pfam mapping
    ct_pfam_map = dict()
    for ct in ct_pfam.index:
        temp = list(ct_pfam.loc[ct, :].unique())
        idx = [pfam_map.loc[x, 0] for x in temp if type(x) is str and x in pfam_map.index]
        ct_pfam_map[ct] = idx
    ## Include domain features for the data
    # create domain features
    for dom in ct_pfam_map[ctype]:
        feat_name = "DOM_{}".format(dom)
        data[feat_name] = [0] * data.shape[0]
    # update values in domain features
    for idx, dom in zip(data.index, data.DOMAINS):
        if type(dom) == str:
            pfam = [x.split(":")[1] for x in dom.split(";") if "Pfam_domain:" in x and x.split(":")[1] in ct_pfam_map[ctype]]
            if pfam != []:
                for pfam_id in pfam:
                    feat_name = "DOM_{}".format(pfam_id)
                    data.loc[idx, feat_name] = 1
        
    return data


def getSNV_X(data, sample_list, DATAPATH, feat_num, ctype, lab_type):
    """
    Process the data to generate features using the data given and 
    remove datapoints with missing data.

    Parameters
    ----------
    data : DataFrame
        .
    sample_list : list
        DESCRIPTION.
    DATAPATH : str
        Path to PIVOT data folder.
    feat_num : str
        "All" or "Some" feature sets.
    ctype : str
        TCGA cancer type.
    lab_type : str
        Labelling strategy to be used.

    Returns
    -------
    X : DataFrame
        Features to be given to PIVOT and the labels.
    X_meta : DataFrame
        Meta data corresponding to the DataFrame.

    """
    # get split data
    X = data[data.Tumor_Sample_Barcode.isin(sample_list)]
    # add labels
    if lab_type == "civic":
        X = getCivicLabels(X, DATAPATH)
    if lab_type == "martellotto":
        X = getMartelottoLabels(X, DATAPATH)
    if lab_type == "cgc":
        X = getCGCLabels(X, DATAPATH)
    if lab_type == "bailey":
        X = getBaileyLabels(X, DATAPATH, ctype)
    # keep labelled data
    X = X[X.Label != "Unlabelled"]
   
    # drop nan rows or columns 
    if feat_num == "all":  
        X.dropna(axis=0, inplace=True)
    if feat_num == "some":
        X.dropna(axis=1, inplace=True)
        X.dropna(axis=0, inplace=True)

    # drop meta-data columns
    X.index = ["{};{}".format(samp[:16], gene) for samp, gene in zip(X.Tumor_Sample_Barcode, X.Hugo_Symbol)]
    X_meta = X.loc[:, ['Hugo_Symbol',"Tumor_Sample_Barcode", "Chr", "Start",
                "End", "Ref", "Alt", "Gene", "Variant_Classification", "Label"]]
    X_meta.index = X.index
    d_cols = [x for x in X.columns if x in ['Hugo_Symbol',"Tumor_Sample_Barcode", "Chr", "Start",
                "End", "Ref", "Alt", "Gene", "Variant_Classification", "DOMAINS"]]
    X = X.drop(d_cols, axis=1)
  
    return (X, X_meta)


def feat_importance(model, model_obj, feat_list):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    model_obj : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    data_plot = pd.DataFrame()
    data_plot["Features"] = feat_list
    if model == "balRF":
        data_plot.loc[:, "Importance"] = model_obj.best_estimator_.feature_importances_
    elif model == "EasyEns" or model == "BalBag":
        data_plot.loc[:, "Importance"] = np.mean([est.steps[1][1].feature_importances_ for est in model_obj.best_estimator_.estimators_], axis=0)
    # elif model == "BalBag":
    #     data_plot["Importance"] = np.mean([tree.feature_importances_ for tree in model_obj.best_estimator_.estimators_], axis=0)

    data_plot = data_plot.sort_values(['Importance'], ascending=False).reset_index(drop=True)
    sns.barplot(x="Importance", y="Features", data=data_plot.iloc[:20,:])
    # sns.barplot(x="Features", y="Importance", data=data_plot.iloc[:20,:])

    plt.gcf().set_size_inches(6,6)
    if model == "balRF":
        plt.title('Feature importance (Balanced Random Forest)')
    elif model == "EasyEns":
        plt.title('Feature importance (Easy Ensemble)')
    elif model == "BalBag":
        plt.title('Feature importance (Balanced Bagging)')

    plt.tight_layout()
    # plt.xticks(rotation=90)
    # fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.50)

    fname = "FeatImp_{}.png".format(model)
    plt.savefig(fname, dpi=300)
    plt.close()
    return data_plot


def plotROC(n_classes, y_test, y_prob, model):
    """
    Plots ROC

    Parameters
    ----------
    n_classes : list
        Set of classes.
    y_test : list
        True labels.
    y_prob : list
        Predicted probabilities.
    model : object
        Model used for training.

    Returns
    -------
    None.

    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_all = []
    for i, c in enumerate(n_classes):
        y_t = [1 if x == c else 0 for x in y_test]
        y_all.append(y_t)
        fpr[i], tpr[i], _ = roc_curve(y_t, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(np.array(y_all).transpose().ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i, c in enumerate(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i, c in enumerate(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= len(n_classes)
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    # plt.figure()
    plt.figure(figsize=(7, 8))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    lw = 2
    for (i, c), color in zip(enumerate(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(c, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if model == "balRF":
        plt.title('Receiver operating characteristic for Balanced Random Forest')
    elif model == "EasyEns":
        plt.title('Receiver operating characteristic for Easy Ensemble')
    elif model == "BalBag":
        plt.title('Receiver operating characteristic for Balanced Bagging')
    plt.legend(loc=(0, -.38), prop=dict(size=10))
    # plt.legend(loc="lower right")
    fname = "ROCplot_{}.png".format(model)
    plt.savefig(fname, dpi=300)
    plt.close()
    return


def plotPRC(n_classes, y_test, y_prob, model):
    """
    Plots PRC

    Parameters
    ----------
    n_classes : list
        Set of classes.
    y_test : list
        True labels.
    y_prob : list
        Predicted probabilities.
    model : object
        Model used for training.

    Returns
    -------
    None.

    """
    precision = dict()
    recall = dict()
    average_precision = dict()
    y_all = []
    # y_mat = []
    for i, c in enumerate(n_classes):
        y_t = [1 if x == c else 0 for x in y_test]
        y_all.append(y_t)
        # y_mat.append(y_t)
        precision[i], recall[i], _ = precision_recall_curve(y_t,
                                                        y_prob[:, i])
        average_precision[i] = average_precision_score(y_t, y_prob[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(np.array(y_all).transpose().ravel(),
             y_prob.ravel())
    # y_mat = np.array(np.transpose(np.matrix(y_mat)))
    average_precision["micro"] = average_precision_score(np.array(y_all).transpose(), y_prob,
                                                     average="micro")
    
    # plot
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    for (i, c), color in zip(enumerate(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(c, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if model == "balRF":
        plt.title('Precision-Recall curve using Balanced Random Forest')
    elif model == "EasyEns":
        plt.title('Precision-Recall curve using Easy Ensemble')
    elif model == "BalBag":
        plt.title('Precision-Recall curve using Balanced Bagging')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=10))
    
    fname = "PRplot_{}.png".format(model)
    plt.savefig(fname, dpi=300)
    plt.close()
    return


if __name__ == "__main__":
    # PATH = "D:/Projects/cTaG2.0"
    # DATAPATH = "D:/Projects/cTaG2.0/data"
    
    PATH = "/data/malvika/cTaG2.0"
    DATAPATH = "/data/malvika/cTaG2.0/data"
    
    n_threads = 20 
    ctype="LUAD"
    folderpath = "/output/GDC_{}/SNV_".format(ctype)
    os.makedirs(PATH + folderpath, exist_ok=True)
    random.seed(3)
    tr_frac = 0.7
    
    # Load data
    os.chdir(DATAPATH + "/GDC_{}/SNV".format(ctype))
    fname="{}_snv.tsv".format(ctype)
    data_snp = pd.read_csv(fname, sep="\t", header=0)
    
    # Set paths
    path_domains = DATAPATH + "/domains/pfam"
    
    # split samples into train test
    train = random.sample(list(data_snp.Tumor_Sample_Barcode.unique()), 
                          int(tr_frac * len(data_snp.Tumor_Sample_Barcode.unique())))
    test = list(set(data_snp.Tumor_Sample_Barcode.unique()).difference(train))
     
    # Generate features
    data = getSNVFeatures(data_snp, ctype, path_domains)
    
    for lab_type in ["civic", "martellotto", "cgc", "bailey"]:
        for feat_num in ["all", "some"]:
            models = ["balRF", "EasyEns","BalBag"]
            for model in models:
                tr_data = data.copy()
                ts_data = data.copy()
                (tr_data, tr_meta) = getSNV_X(tr_data, train, DATAPATH, feat_num, ctype)
                (ts_data, ts_meta) = getSNV_X(ts_data, test, DATAPATH, feat_num, ctype)
    
                samp_tr = len(tr_meta.Tumor_Sample_Barcode.unique())
                samp_ts = len(ts_meta.Tumor_Sample_Barcode.unique())
                
                if len(tr_data.Label.unique()) > 1:
                    # Build models
                    if model == "balRF":
                        param_rfc = {'max_features': [30, 40, 50], 'max_depth': [2, 3, 4],
                                     'criterion': ['gini', 'entropy'], 'n_estimators': [50, 100, 200, 500]}
                        clf = BalancedRandomForestClassifier(n_jobs=2, random_state=0)
                    if model == "EasyEns":
                        param_rfc = {'n_estimators':  [10, 20, 30, 40, 100, 200, 500]}
                        clf = EasyEnsembleClassifier(n_jobs=2, random_state=0)
                    if model == "BalBag":
                        param_rfc = {'max_features': [1.0],
                                     'n_estimators': [10, 20, 30, 40, 100, 200, 500]}
                        clf = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                                        n_jobs=2,
                                                        sampling_strategy='auto',
                                                        replacement=False,
                                                        random_state=0)
                
                    gs = GridSearchCV(estimator=clf, param_grid=param_rfc,
                                      scoring='f1_weighted', cv=5, verbose=1, 
                                      n_jobs=n_threads)
                    gs = gs.fit(tr_data.drop(["Label"], axis=1), tr_data.Label)
                    
                    
                    # Get predictions
                    tr_pred = gs.predict(tr_data.drop(["Label"], axis=1))
                    ts_pred = gs.predict(ts_data.drop(["Label"], axis=1))
                    
                    # calculate metrics and print to o/p file
                    n_classes = sorted(tr_data.Label.unique())
                    f1_tr = f1_score(tr_data.Label, tr_pred, average=None, labels=n_classes)
                    f1_ts = f1_score(ts_data.Label, ts_pred, average=None, labels=n_classes)
                    p_tr = precision_score(tr_data.Label, tr_pred, average=None, labels=n_classes)
                    p_ts = precision_score(ts_data.Label, ts_pred, average=None, labels=n_classes)
                    r_tr = recall_score(tr_data.Label, tr_pred, average=None, labels=n_classes)
                    r_ts = recall_score(ts_data.Label, ts_pred, average=None, labels=n_classes)
                    a_tr = accuracy_score(tr_data.Label, tr_pred)
                    a_ts = accuracy_score(ts_data.Label, ts_pred)
                
                    os.makedirs(PATH + folderpath +"/{}_{}_{}".format(lab_type, feat_num, model), exist_ok=True)
                    os.chdir(PATH + folderpath + "/{}_{}_{}".format(lab_type, feat_num, model))
        
                    # Save model
                    os.chdir(PATH + folderpath +"/{}_{}_{}".format(lab_type, feat_num, model))
                    featFName = "model_{}_{}_{}.pkl".format(lab_type, feat_num, model)
                    with open(featFName, 'wb') as f:
                        pickle.dump(gs, f)
                        
                    # Plots
                    plotPRC(gs.classes_, ts_data.Label, 
                            gs.predict_proba(ts_data.drop(["Label"], axis=1)),
                            model)
                    plotROC(gs.classes_, ts_data.Label,
                            gs.predict_proba(ts_data.drop(["Label"], axis=1)),
                            model)
        
                    # Plot feature importance
                    feat_data = feat_importance(model, gs, tr_data.columns[:-1])
                    fname = "Feature_imp.tsv"
                    feat_data.to_csv(fname, header=True, index=False, sep="\t")
                    
                    # Save predictions
                    tr = tr_meta.copy()
                    tr["Predicted label"] = tr_pred
                    temp = gs.predict_proba(tr_data.drop(["Label"], axis=1))
                    for idx, p_class in enumerate(gs.classes_):
                        col_name = "Probability_{}".format(p_class)
                        tr[col_name] = temp[:,idx]
                    tr.to_csv("train_predictions.tsv", header=True, index=True,
                              sep="\t")
                    
                    ts = ts_meta.copy()
                    ts["Predicted label"] = ts_pred
                    temp = gs.predict_proba(ts_data.drop(["Label"], axis=1))
                    for idx, p_class in enumerate(gs.classes_):
                        col_name = "Probability_{}".format(p_class)
                        ts[col_name] = temp[:,idx]
                    ts.to_csv("test_predictions.tsv", header=True, index=True,
                              sep="\t")
        
                    # Save metrics
                    fname = "metrics_{}_{}_{}.txt".format(lab_type, feat_num, model)
                    with open(fname, 'w') as f:
                        f.write("Data trained on {}\n".format(lab_type))
                        f.write("Features trained on {}\n".format(feat_num))
                        f.write("Number of features = {}\n".format(tr_data.shape[1]-1))
                        f.write("Features\t{}\n".format(tr_data.drop(["Label"], axis=1).columns))
                        f.write("\n\n\n\t")
                        f.write("\t".join(list(gs.classes_)))
                        f.write("\nTrain\t")
                        f.write("\t".join([str(list(tr_data.Label).count(x)) for x in list(gs.classes_)]))
                        f.write("\nTest\t")
                        f.write("\t".join([str(list(ts_data.Label).count(x)) for x in list(gs.classes_)]))
                        f.write("\n\nTotal number samples (Train) = {}\n".format(samp_tr))
                        f.write("\nTotal number samples (Test) = {}\n\n".format(samp_ts))
                        f.write("Model used = {}\n".format(model))
                        f.write("Best estimator = {}\n".format(gs.best_estimator_))
                        f.write("Best parameter = {}\n".format(gs.best_params_))
                        f.write("Best score = {}\n".format(gs.best_score_))
                        f.write("\tTraining\t\t\tTest\n")
                        f.write("\t")
                        f.write("\t".join(list(gs.classes_)))
                        f.write("\t")
                        f.write("\t".join(list(gs.classes_)))
                        f.write("\n")
                        f.write("Accuracy\t{:1.4f}".format(a_tr))
                        for i in range(len(gs.classes_)):
                            f.write("\t")
                        f.write("{:1.4f}".format(a_ts))
                        f.write("\n")
                        f.write("F1 score\t")
                        for i in f1_tr:
                            f.write("{:1.4f}\t".format(i))
                        for i in f1_ts:
                            f.write("{:1.4f}\t".format(i))               
                        f.write("\n")
                        f.write("Precision\t")
                        for i in p_tr:
                            f.write("{:1.4f}\t".format(i))
                        for i in p_ts:
                            f.write("{:1.4f}\t".format(i))
                        f.write("\n")
                        f.write("Recall\t")
                        for i in r_tr:
                            f.write("{:1.4f}\t".format(i))
                        for i in r_ts:
                            f.write("{:1.4f}\t".format(i))
                        f.write("\n")
                else:
                    os.makedirs(PATH + folderpath +"/{}_{}_{}".format(lab_type, feat_num, model), exist_ok=True)
                    os.chdir(PATH + folderpath + "/{}_{}_{}".format(lab_type, feat_num, model))
                    # Save metrics
                    fname = "metrics_{}_{}_{}.txt".format(lab_type, feat_num, model)
                    with open(fname, 'w') as f:
                        f.write("Data trained on {}\n".format(lab_type))
                        f.write("Features trained on {}\n".format(feat_num))
                        f.write("Number of features = {}\n".format(tr_data.shape[1]-1))
                        f.write("Features\t{}\n".format(tr_data.drop(["Label"], axis=1).columns))
                        f.write("\n\n\n\t")
                        f.write("\t".join(list(tr_data.Label.unique())))
                        f.write("\nTrain\t")
                        f.write("\t".join([str(list(tr_data.Label).count(x)) for x in list(tr_data.Label.unique())]))
                        f.write("\nTest\t")
                        f.write("\t".join([str(list(ts_data.Label).count(x)) for x in list(tr_data.Label.unique())]))
                        f.write("\n\nTotal number samples (Train) = {}\n".format(samp_tr))
                        f.write("\nTotal number samples (Test) = {}\n\n".format(samp_ts))
               
