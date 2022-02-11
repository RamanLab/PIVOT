# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:08:45 2021

@author: MalvikaS

Build classifier on RNA seq data
"""

# Import 
import os
import pandas as pd
import random
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score
import pickle
import multiprocessing as mp
import logging
import glob
import snv_classifier as snv


def getCnvNeighFeat(data_all, data_rna, G, neighbors, norm = False):
    """
    Function for getting neighbourhood features. Calculates the sum of logFC
    across the different neighbours of the gene.
    
    To normalize for number of neighbours with change in fold change (FC), the 
    norm parameter need to be True.
    
    Function considers all genes as neighbours as passed in arguments. These
    neighbours are assumed to be n hops away. 

    Parameters
    ----------
    data_all : DataFrame
        DataFrame containing the fold change data. Should contain column 
        "logFC".
    data_rna : DataFrame
        DataFrame containing the fold change data. Should contain column 
        "logFC".
    G : networkx graph object
        Undirected graph of protein protein interactions.
    neighbors : dict
        Dictionary with gene as keys and list of neighbours as value.
    norm : bool, optional
        Whether to normalise for neighbour up or down-regulated. The default
        is False.

    Returns
    -------
    feat : list
        list of values for each gene. 

    """
    # Get sum of neighbour FC
    feat = []
    for gene, cnv in zip(data_all["Hugo_Symbol"].values, data_all["CNV"].values):
        val = 0
        count = 0
        if gene in G.nodes and cnv != 0:
            for neigh in neighbors[gene]:
                if neigh in data_rna["genes"].values and abs(data_rna.loc[neigh, "logFC"]) > 2:
                   val = val + abs(data_rna.loc[neigh, "logFC"])
                   count = count + 1
        if norm and count != 0:
            feat.append((val * cnv) / count)
        else:
            feat.append(val)
    return feat



def compareCols(list1, list2, value):
    """
    Function for handelling nan values by comparing two columns

    Parameters
    ----------
    list1 : list
        DESCRIPTION.
    list2 : list
        DESCRIPTION.
    value : str
        Value to appended based on values on two columns

    Returns
    -------
    None.

    """
    temp = []
    for x, y in zip(list1, list2):
        if x is np.nan and y is not np.nan:
            temp.append(y)
        elif y is np.nan and x is not np.nan:
            temp.append(x)
        elif x is np.nan and y is np.nan:
            temp.append(np.nan)
        elif x == value and y != value:
            temp.append(y)
        elif y == value and x != value:
            temp.append(x)
        elif x == y:
            temp.append(x)
        else:
            temp.append(value)
    return temp


def getNeighFeat(genes, data_rna_all, G, neighbors, norm = False):
    """
    Function for getting neighbourhood features. Calculates the sum of logFC
    across the different neighbours of the gene.
    
    To normalize for number of neighbours with change in fold change (FC), the 
    norm parameter need to be True.
    
    Function considers all genes as neighbours as passed in arguments. These
    neighbours are assumed to be n hops away. 

    Parameters
    ----------
    genes : list
        List of genes to be considered.
    data_rna_all : DataFrame
        DataFrame containing the fold change data. Should contain column 
        "logFC".
    G : networkx graph object
        Undirected graph of protein protein interactions.
    neighbors : dict
        Dictionary with gene as keys and list of neighbours as value.
    norm : bool, optional
        Whether to normalise for neighbour up or down-regulated. The default
        is False.

    Returns
    -------
    feat : list
        list of values for each gene. 

    """
    # Get sum of neighbour FC
    feat = []
    for gene in genes:
        val = 0
        count = 0
        if gene in G.nodes:
            for neigh in neighbors[gene]:
                if neigh in data_rna_all["genes"].values and abs(data_rna_all.loc[neigh, "logFC"]) > 2:
                   val = val + abs(data_rna_all.loc[neigh, "logFC"])
                   count = count + 1
        if norm and count != 0:
            feat.append(val / count)
        else:
            feat.append(val)
    return feat


def getMultiFeatures(args):
    """
    Function for generating RNA features using egdeR results as well as 
    network.
    
    Generates features using parallel process.

    Parameters
    ----------
    datapath : str
        Complete folder path where RNA processed files are saved. Each file 
        contains DEGs for each patient. Must contain columns "genes", 'logFC'.
    file : str
        File name to be read and features generated. A default ".tsv" is 
        included to the file name.
    ctype : str
        Cancer-type.
    path_network : str
        Complete path to where network data is stored as pkl files.
    n : int, optional
        Number hops to be considered while defining neighbours. 
        The default is 1.

    Returns
    -------
    DataFrame
        DataFrame containing all the RNA features to be used for model building.

    """
    (file, [data, rnapath, ctype, G, deg, bc, cc, neighbors,
                 n]) = args
    data_all = data.copy()
    data_all = data_all[data_all["Tumor_Sample_Barcode"] == file]
    # print(file)
 
    # Load RNA data
    # Read RNA file
    os.chdir(rnapath)
    rna_flist = glob.glob("*.tsv")
    if file+".tsv" not in rna_flist:
        return None
    data_rna_all = pd.read_csv(file+".tsv", header=0, index_col=0, sep="\t")
    data_rna = data_rna_all[data_rna_all["genes"].isin(data_all["Hugo_Symbol"])]
    
    # Get degree
    temp =[deg[gene] if gene in G.nodes else 0 for gene in data_rna["genes"].values]
    data_rna.loc[:, "Degree"] = temp
    # Get closeness centrality
    temp =[cc[gene] if gene in G.nodes else 0 for gene in data_rna["genes"].values]
    data_rna.loc[:, "Closeness_centrality"] = temp
    # Get betweeness centrality
    temp =[bc[gene] if gene in G.nodes else 0 for gene in data_rna["genes"].values]
    data_rna.loc[:, "Betweeness_centrality"] = temp
    
    # Get FC x degree
    temp =[fc * d if abs(fc) >2 else 0 for fc, d in zip(data_rna["logFC"].values,
                                                        data_rna["Degree"].values)]
    data_rna.loc[:, "FC_Degree"] = temp
    # Get FC x Closeness_centrality
    temp =[fc * c if abs(fc) >2 else 0 for fc, c in zip(data_rna["logFC"].values,
                                                        data_rna["Closeness_centrality"].values)]
    data_rna.loc[:, "FC_Closeness_centrality"] = temp
    # Get FC x Betweeness_centrality
    temp =[fc * b if abs(fc) >2 else 0 for fc, b in zip(data_rna["logFC"].values,
                                                        data_rna["Betweeness_centrality"].values)]
    data_rna.loc[:, "FC_Betweeness_centrality"] = temp
    
    # Get sum of FC of neighbours
    data_rna.loc[:, "neigh_FC"] = getNeighFeat(data_rna["genes"].values, data_rna_all, G, neighbors, norm = False)
    # data_rna.loc[data_rna.index, "neigh_FC"] = data_rna_all.loc[data_rna.index, "neigh_FC"].values
    # Get normalized sum of FC of neighbours
    data_rna.loc[:, "neigh_FC"] = getNeighFeat(data_rna["genes"].values, data_rna_all, G, neighbors, norm = True)
    # data_rna.loc[data_rna.index, "neigh_normFC"] = data_rna_all.loc[data_rna.index, "neigh_normFC"].values

    # merge rna and other data
    data_rna.loc[:, "Hugo_Symbol"] = data_rna.loc[:, "genes"]
    data_all = pd.merge(data_all, data_rna, on='Hugo_Symbol', how='left')
    
    # Get CNV change in neighbour FC
    data_all.loc[:, "neigh_CNV_FC"] = getCnvNeighFeat(data_all, data_rna_all, G, neighbors, norm = False)
    # Get normalized sum of FC of neighbours
    data_all.loc[:, "neigh_CNV_normFC"] = getCnvNeighFeat(data_all, data_rna_all, G, neighbors, norm = True)
    
    # Assign indices
    data_all.index = ["{};{}".format(file, gene) for gene in data_all.loc[:, "Hugo_Symbol"]]
    # data_rna["Tumor_Sample_Barcode"] = [file] * len(data_rna)
    return data_all


def get_multiX(data_snv, rnapath, data_cnv, data_miRNA, sample_list, DATAPATH,
               ctype, lab_type, n_threads=1, train=False):
    """
    Get X for multi-omic prediction. The required columns are retained and all
    other rows and columns dropped. This function also labels the data for 
    building models. RNA files are read at this stage and data points 
    correspondng to labelled data are retained.

    Parameters
    ----------
    data_snv : TYPE
        DESCRIPTION.
    rnapath : TYPE
        DESCRIPTION.
    data_cnv : TYPE
        DESCRIPTION.
    data_miRNA : TYPE
        DESCRIPTION.
    sample_list : list
        List of tumour samples to be retained.
    DATAPATH : str
        Complete path to SNV data for the samples and other data for different
        laabelling techniques.
    ctype : str
        Cancer-type.
    lab_type : str
        Labelling stratergy to be used.
    n_threads : int
        Number of threads to be used. Default 1.
    train : str
        If True removes rows Unlabelled. Default False.

    Returns
    -------
    data : DataFrame
        DataFrame containing feature matrix to be trained on and labels.
    data_meta : DataFrame
        DataFrame containing mata data for the feature matrix.
    """
    # Process SNV data (for labelling)
    data_snv = data_snv.copy()
    data_snv = data_snv[data_snv.Tumor_Sample_Barcode.isin(sample_list)]
    data_snv["samp_gene"] = ["{};{}".format(samp[:16], gene) for samp, gene in zip(data_snv.Tumor_Sample_Barcode, data_snv.Hugo_Symbol)]

    # Add labels to SNV data
    if lab_type == "civic":
        data_snv = snv.getCivicLabels(data_snv, DATAPATH)
    if lab_type == "martellotto":
        data_snv = snv.getMartelottoLabels(data_snv, DATAPATH)    
    if lab_type == "cgc":
        data_snv = snv.getCGCLabels(data_snv, DATAPATH)
    if lab_type == "bailey":
        data_snv = snv.getBaileyLabels(data_snv, DATAPATH, ctype)
    # TODO
    logging.info("Got snv labels")

    # Process CNV data (for labelling)
    data_cnv = data_cnv.copy()
    data_cnv["Tumor_Sample_Barcode"] = [idx.split(";")[0] for idx in data_cnv.index]
    data_cnv = data_cnv[data_cnv.Tumor_Sample_Barcode.isin(sample_list)]
    data_cnv["Hugo_Symbol"] = [x.split(";")[1] for x in data_cnv.index]
    data_cnv["samp_gene"] = list(data_cnv.index)

    # Add labels to CNV data
    if lab_type == "civic":
        data_cnv["Label"] = ["Unlabelled"] * len(data_cnv)
    if lab_type == "martellotto":
        data_cnv["Label"] = ["Unlabelled"] * len(data_cnv)
    if lab_type == "cgc":
        data_cnv = snv.getCGCLabels(data_cnv, DATAPATH)
    if lab_type == "bailey":
        data_cnv = snv.getBaileyLabels(data_cnv, DATAPATH, ctype)
    # TODO
    logging.info("Got CNV labels")
    
    # Merge data
    data = pd.merge(data_snv, data_cnv, on='samp_gene', how='outer')
    # TODO
    logging.info("merged data")

    # Get features 
    path_domains = DATAPATH + "/domains/pfam"
    data = snv.getSNVFeatures(data, ctype, path_domains)
    # TODO
    logging.info("Got snv features")

    # Concat labels and fill missing data
#     if  lab_type == "cgc" or lab_type == "bailey":
    data["Label"] = compareCols(data.Label_x, data.Label_y, "Unlabelled")
    data = data.drop(["Label_x", "Label_y"], axis=1)
    data["Hugo_Symbol"] = compareCols(data.Hugo_Symbol_x, data.Hugo_Symbol_y,
                                      np.nan)
    data = data.drop(["Hugo_Symbol_x", "Hugo_Symbol_y"], axis=1)
    data["Tumor_Sample_Barcode"] = compareCols(data["Tumor_Sample_Barcode_x"].values,
                                               data["Tumor_Sample_Barcode_y"].values,
                                               np.nan)
    data = data.drop(["Tumor_Sample_Barcode_x", "Tumor_Sample_Barcode_y"],
                     axis=1)
    
    if train:
        # Remove duplicates and keep labelled data_snp
        data = data[data.Label != "Unlabelled"]
    
    # fill na in CNV column
    data["CNV"].fillna(0, inplace=True)
    # TODO
    logging.info("Dropped columns and filled na")
    
    # Add miRNA features
    for miRNA in data_miRNA.index.unique():
        print(miRNA)
        # data[miRNA] = [data_miRNA[data_miRNA["Sample ID"] == samp].loc[miRNA, "reads_per_million_miRNA_mapped"] if samp in list(data_miRNA["Sample ID"]) else np.nan for samp in data.Tumor_Sample_Barcode]
        data[miRNA] = [0] * len(data)
        for samp in data.Tumor_Sample_Barcode.unique():
            if samp in data_miRNA["Sample ID"].values:
                idx = data[data["Tumor_Sample_Barcode"] == samp].index
                data.loc[idx, miRNA] = [data_miRNA[data_miRNA["Sample ID"] == samp].loc[miRNA, "reads_per_million_miRNA_mapped"]] * len(idx)
    # TODO
    logging.info("miRNA features")

    # load data
    path_network = DATAPATH + "/network"
    n = 1
    # Load network
    os.chdir(path_network) 
    with open("string_graph.pkl", "rb") as f:
        G = pickle.load(f)
    with open("string_degree.pkl", "rb") as f:
        deg = pickle.load(f)
    with open("string_bc.pkl", "rb") as f:
        bc = pickle.load(f)
    with open("string_cc.pkl", "rb") as f:
        cc = pickle.load(f)
    with open("string_neigh_{}.pkl".format(n), "rb") as f:
        neighbors = pickle.load(f)
    
#     data_all = [None] * len(sample_list)
       
    args_lst = [(file, [data, rnapath, ctype, G, deg, bc, cc, neighbors,
                 n]) for file in sample_list]
    pool = mp.Pool(processes = n_threads)
    data_all = pool.map(getMultiFeatures, args_lst)
    pool.close()
    pool.join()
    # TODO
    logging.info("concat RNA features")
    data_all = pd.concat(data_all)

    # TODO
    logging.info("included RNA features")

    # fill na 
    data_all["logFC"].fillna(0, inplace=True)
    data_all["logCPM"].fillna(0, inplace=True)
    data_all["Degree"].fillna(0, inplace=True)
    data_all["Closeness_centrality"].fillna(0, inplace=True)
    data_all["FC_Degree"].fillna(0, inplace=True)
    data_all["FC_Closeness_centrality"].fillna(0, inplace=True)
    data_all["FC_Betweeness_centrality"].fillna(0, inplace=True)
    data_all["neigh_FC"].fillna(0, inplace=True)
    data_all["Betweeness_centrality"].fillna(0, inplace=True)
    

    # Define meta-data and drop meta-data columns from RNA data
    drop_cols = ['Gene', 'Chr', 'Start', 'End', 'Ref', 'Alt', 
                 'Variant_Classification', 'DOMAINS', 'samp_gene',
                 'Hugo_Symbol', 'Tumor_Sample_Barcode', 'genes',
                 'unshrunk.logFC', 'PValue', 'FDR']
    data_meta = data_all[['Hugo_Symbol', 'Tumor_Sample_Barcode',
                     'Variant_Classification', 'CNV', 'logFC', 'Degree',
                     'Closeness_centrality', 'Betweeness_centrality', 'Label']]
    data_meta.index = data_all.index
    d_cols = [x for x in data_all.columns if x in drop_cols]
    data_all = data_all.drop(d_cols, axis=1)
    # TODO
    logging.info("returning labelled features")

    return (data_all, data_meta)


def dropNa(data, data_meta, feat_num):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    data_meta : TYPE
        DESCRIPTION.
    feat_num : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    data_all = data.copy()
    X_meta = data_meta.copy()
    data_all["temp"] = list(range(len(data_all)))
    X_meta["temp"] = list(range(len(data_all)))
    # drop nan rows or columns 
    if feat_num == "all":  
        X = data_all.dropna(axis=0, inplace=False)
        X_meta = X_meta[X_meta["temp"].isin(X["temp"])]
    if feat_num == "some":
        X =  data_all.dropna(axis=1, inplace=False)
        X.dropna(axis=0, inplace=True)
        X_meta = X_meta[X_meta["temp"].isin(X["temp"])]
    X.drop(["temp"], axis=1, inplace=True)
    X_meta.drop(["temp"], axis=1, inplace=True)
    return (X, X_meta)


if __name__ == "__main__":
    # Set path
    # PATH = "D:/Projects/cTaG2.0"
    # DATAPATH = "D:/Projects/cTaG2.0/data"
    PATH = "/data/malvika/cTaG2.0"
    DATAPATH = "/data/malvika/cTaG2.0/data"
    
    # Load dependent modules
    os.chdir(PATH + "/code")
    import snp_classifier as snv
    # import rna_classifier as rna
    
    # Define variables. To be set for each run
    n_threads = 20                                      # number of threads
    ctype = "LUAD"                                      # cancer-type
    folderpath = "/output/GDC_{}/multiomic".format(ctype)     # output folder path
    os.makedirs(PATH + folderpath, exist_ok=True)
    random.seed(3)                                      # random seed
    tr_frac = 0.7                                       # train-test fraction
    logging.basicConfig(level=logging.INFO)
    
    # Set paths
    path_network = DATAPATH + "/network"
    path_domains = DATAPATH + "/domains/pfam"

    # Load data
    # SNV
    os.chdir(DATAPATH + "/GDC_{}/SNV".format(ctype))
    fname="{}_snv.tsv".format(ctype)
    data_snv = pd.read_csv(fname, sep="\t", header=0)
    data_snv["Tumor_Sample_Barcode"] = [samp[:16] for samp in data_snv["Tumor_Sample_Barcode"]]
    # RNA
    rnapath = DATAPATH + "/GDC_{}/RNA-seq".format(ctype)
    # CNV
    os.chdir(DATAPATH + "/GDC_{}/CNV".format(ctype))
    fname="{}_cnv.tsv".format(ctype)
    data_cnv = pd.read_csv(fname, sep="\t", header=0, index_col=0)
    # miRNA
    os.chdir(DATAPATH + "/GDC_{}/miRNA".format(ctype))
    fname="{}_miRNA.tsv".format(ctype)
    data_miRNA = pd.read_csv(fname, sep="\t", header=0, index_col=0)
    # TODO
    logging.info("Loaded data")

    # split samples into train test
    train = random.sample(list(data_snv.Tumor_Sample_Barcode.unique()), 
                          int(tr_frac * len(data_snv.Tumor_Sample_Barcode.unique())))
    test = list(set(data_snv.Tumor_Sample_Barcode.unique()).difference(train))
    
    # # TODO
    # # Delete rows below
    # train = train[:100]
    # test = test[:100]

    for lab_type in ["civic", "martellotto", "cgc", "bailey"]:
        # TODO
        logging.info(lab_type)
        # Get train-test data
        (data_train, train_meta) = get_multiX(data_snv, rnapath, data_cnv,
                                           data_miRNA, train, DATAPATH,
                                           ctype, lab_type, 
                                           n_threads=n_threads*2)
        # TODO
        logging.info("Ran get_multiX")

        (data_test, test_meta) = get_multiX(data_snv, rnapath, data_cnv,
                                           data_miRNA, test, DATAPATH,
                                           ctype, lab_type, 
                                           n_threads=n_threads*2)

        for feat_num in ["all", "some"]:
            # TODO
            logging.info(feat_num)
            (tr_data, tr_meta) = dropNa(data_train, train_meta, feat_num)
            (ts_data, ts_meta) = dropNa(data_test, test_meta, feat_num)

            samp_tr = len(tr_meta.Tumor_Sample_Barcode.unique())
            samp_ts = len(ts_meta.Tumor_Sample_Barcode.unique())

            models = ["balRF", "EasyEns","BalBag"]
            for model in models:
                logging.info(model)
                logging.info(tr_data.Label.unique())

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
                    logging.info("Building models")
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

                    logging.info("Saving model and output")
                    # Save model
                    os.chdir(PATH + folderpath +"/{}_{}_{}".format(lab_type, feat_num, model))
                    featFName = "model_{}_{}_{}.pkl".format(lab_type, feat_num, model)
                    with open(featFName, 'wb') as f:
                        pickle.dump(gs, f)
                        
                    # Plots
                    snv.plotPRC(gs.classes_, ts_data.Label, 
                            gs.predict_proba(ts_data.drop(["Label"], axis=1)),
                            model)
                    snv.plotROC(gs.classes_, ts_data.Label,
                            gs.predict_proba(ts_data.drop(["Label"], axis=1)),
                            model)
        
                    # Plot feature importance
                    feat_data = snv.feat_importance(model, gs, tr_data.drop(["Label"], axis=1).columns)
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

