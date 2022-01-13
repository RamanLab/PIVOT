# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:08:45 2021

@author: MalvikaS

Build classifier on RNA seq data
"""

# Import 
import os
import pandas as pd
import glob
import random
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score
import pickle


def getNeighFeat(data_rna, G, neighbors, norm = False):
    """
    Function for getting neighbourhood features. Calculates the sum of logFC
    across the different neighbours of the gene.
    
    To normalize for number of neighbours with change in fold change (FC), the 
    norm parameter need to be True.
    
    Function considers all genes as neighbours as passed in arguments. These
    neighbours are assumed to be n hops away. 

    Parameters
    ----------
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
    for gene in data_rna.genes:
        val = 0
        count = 0
        if gene in G.nodes:
            for neigh in neighbors[gene]:
                if neigh in data_rna.genes and abs(data_rna.loc[neigh, "logFC"]) > 2:
                   val = val + abs(data_rna.loc[neigh, "logFC"])
                   count = count + 1
        if norm and count != 0:
            feat.append(val / count)
        else:
            feat.append(val)
    return feat


def getRNAFeatures(datapath, file, ctype, path_network, n = 1):
    """
    Function for generating RNA features using egdeR results as well as network.

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

    
    # Load RNA data
    # Get sample name
    samp = file
    # Read RNA file
    os.chdir(datapath)
    data_rna = pd.read_csv(file+".tsv", header=0, index_col=0, sep="\t")
    
    # Get degree
    temp =[deg[gene] if gene in G.nodes else 0 for gene in data_rna.genes]
    data_rna["Degree"] = temp
    # Get closeness centrality
    temp =[cc[gene] if gene in G.nodes else 0 for gene in data_rna.genes]
    data_rna["Closeness_centrality"] = temp
    # Get betweeness centrality
    temp =[bc[gene] if gene in G.nodes else 0 for gene in data_rna.genes]
    data_rna["Betweeness_centrality"] = temp
    
    # Get FC x degree
    temp =[fc * d if abs(fc) >2 else 0 for fc, d in zip(data_rna.logFC, data_rna.Degree)]
    data_rna["FC_Degree"] = temp
    # Get FC x Closeness_centrality
    temp =[fc * c if abs(fc) >2 else 0 for fc, c in zip(data_rna.logFC, data_rna.Closeness_centrality)]
    data_rna["FC_Closeness_centrality"] = temp
    # Get FC x Betweeness_centrality
    temp =[fc * b if abs(fc) >2 else 0 for fc, b in zip(data_rna.logFC, data_rna.Betweeness_centrality)]
    data_rna["FC_Betweeness_centrality"] = temp
    
    # Get sum of FC of neighbours
    data_rna["neigh_FC"] = getNeighFeat(data_rna, G, neighbors, norm = False)
    # Get normalized sum of FC of neighbours
    data_rna["neigh_normFC"] = getNeighFeat(data_rna, G, neighbors, norm = True)
    
    # Assign indices
    data_rna.index = ["{};{}".format(samp, gene) for gene in data_rna.genes]
    data_rna["Tumor_Sample_Barcode"] = [samp] * len(data_rna)
    return data_rna


def getRNA_X(sample_list, DATAPATH, ctype, lab_type):
    """
    Get X for RNA. The required columns are retained and all other rows and 
    columns dropped. This function also labels the data for building models.

    Parameters
    ----------
    sample_list : list
        List of tumour samples to be retained.
    DATAPATH : str
        Complete path to SNV data for the samples and other data for different
        laabelling techniques.
    ctype : str
        Cancer-type.
    lab_type : str
        Labelling stratergy to be used.

    Returns
    -------
    data : DataFrame
        DataFrame containing feature matrix to be trained on and labels.
    data_meta : DataFrame
        DataFrame containing mata data for the feature matrix.

    """
    # Load SNV data (for labelling)
    os.chdir(DATAPATH + "/GDC_{}/SNV".format(ctype))
    fname="{}_snv.tsv".format(ctype)
    snv_lab = pd.read_csv(fname, sep="\t", header=0)
    snv_lab.Tumor_Sample_Barcode = [samp[:16] for samp in snv_lab.Tumor_Sample_Barcode]
    snv_lab = snv_lab[snv_lab.Tumor_Sample_Barcode.isin(sample_list)]
    snv_lab.index = ["{};{}".format(samp[:16], gene) for samp, gene in zip(snv_lab.Tumor_Sample_Barcode, snv_lab.Hugo_Symbol)]
    
    # Add labels
    if lab_type == "civic":
        snv_lab = snv.getCivicLabels(snv_lab, DATAPATH)
    if lab_type == "martellotto":
        snv_lab = snv.getMartelottoLabels(snv_lab, DATAPATH)    
    if lab_type == "cgc":
        snv_lab = snv.getCGCLabels(snv_lab, DATAPATH)
    if lab_type == "bailey":
        snv_lab = snv.getBaileyLabels(snv_lab, DATAPATH, ctype)

    # Remove duplicates and keep labelled data_snp
    snv_lab = snv_lab[snv_lab.Label != "Unlabelled"]
    snv_lab = snv_lab[~snv_lab.index.duplicated()]
    
    # load data
    path_network = DATAPATH + "/network"
    data = [None] * len(sample_list)
    datapath = DATAPATH + "/GDC_{}/RNA-seq".format(ctype) 
    for idx, file in enumerate(sample_list):
        temp = getRNAFeatures(datapath, file, ctype, path_network, n=1)
        # Assign labels to RNA data
        temp["Label"] = [snv_lab.loc[idx, "Label"] if idx in snv_lab.index else "Unlabelled" for idx in temp.index]
        temp = temp[temp["Label"] != "Unlabelled"]
        # Drop nan rows 
        data[idx] = temp.dropna(axis=0)

    # Concat data
    data = pd.concat(data)
    
    # Define meta-data and drop meta-data columns from RNA data
    data_meta = data[['genes', 'Tumor_Sample_Barcode', 'Label']]
    data_meta.index = data.index
    d_cols = [x for x in data.columns if x in ['genes', 'unshrunk.logFC',
                                                   'PValue', 'FDR', 
                                                   'Tumor_Sample_Barcode']]
    data = data.drop(d_cols, axis=1)
  
    return (data, data_meta)


if __name__ == "__main__":
    # Set path
    # PATH = "D:/Projects/cTaG2.0"
    # DATAPATH = "D:/Projects/cTaG2.0/data"
    PATH = "/data/malvika/cTaG2.0"
    DATAPATH = "/data/malvika/cTaG2.0/data"

    # Load dependent modules
    os.chdir(PATH + "/code")
    import snp_classifier as snv

    # Define variables. To be set for each run
    n_threads = 20                                      # number of threads
    ctype = "LUAD"                                      # cancer-type
    folderpath = "/output/GDC_{}/RNA".format(ctype)     # output folder path
    os.makedirs(PATH + folderpath, exist_ok=True)
    random.seed(3)                                      # random seed
    tr_frac = 0.7                                       # train-test fraction

    # Set paths
    path_network = DATAPATH + "/network"
    datapath = DATAPATH + "/GDC_{}/RNA-seq".format(ctype)

    # split samples into train test
    os.chdir(datapath) 
    uni_samples = [file[:-4] for file in glob.glob("*.tsv")]
    train = random.sample(uni_samples, int(tr_frac * len(uni_samples)))
    test = list(set(uni_samples).difference(train))

    for lab_type in ["civic", "martellotto", "cgc", "bailey"]:
        # Get train-test data
        (tr_data, tr_meta) = getRNA_X(train, DATAPATH, ctype, lab_type)
        (ts_data, ts_meta) = getRNA_X(test, DATAPATH, ctype, lab_type)
    
        # Get number of samples
        samp_tr = len(tr_meta.Tumor_Sample_Barcode.unique())
        samp_ts = len(ts_meta.Tumor_Sample_Barcode.unique())

        models = ["balRF", "EasyEns","BalBag"]
        for model in models:
            # Check if more than one label for training
            if len(tr_data.Label.unique()) > 1:
                # Define models
                if model == "balRF":
                    param_rfc = {'max_features': [10, 20, 30, 40],
                                 'max_depth': [2, 3, 4],
                                 'criterion': ['gini', 'entropy'],
                                 'n_estimators': [50, 100, 200, 500]}
                    clf = BalancedRandomForestClassifier(n_jobs=2,
                                                         random_state=0)
                if model == "EasyEns":
                    param_rfc = {'n_estimators':  [10, 20, 30, 40, 100, 200,
                                                   500]}
                    clf = EasyEnsembleClassifier(n_jobs=2, random_state=0)
                if model == "BalBag":
                    param_rfc = {'max_features': [1.0],
                                 'n_estimators': [10, 20, 30, 40, 100, 200, 500]}
                    clf = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                                    n_jobs=2,
                                                    sampling_strategy='auto',
                                                    replacement=False,
                                                    random_state=0)
                
                # Fit models
                gs = GridSearchCV(estimator=clf, param_grid=param_rfc,
                                  scoring='f1_weighted', cv=5, verbose=1, 
                                  n_jobs=n_threads)
                gs = gs.fit(tr_data.drop(["Label"], axis=1), tr_data.Label)
                        
                # Get predictions
                tr_pred = gs.predict(tr_data.drop(["Label"], axis=1))
                ts_pred = gs.predict(ts_data.drop(["Label"], axis=1))
                
                # Calculate metrics and print to o/p file
                n_classes = sorted(tr_data.Label.unique())
                f1_tr = f1_score(tr_data.Label, tr_pred, average=None,
                                 labels=n_classes)
                f1_ts = f1_score(ts_data.Label, ts_pred, average=None,
                                 labels=n_classes)
                p_tr = precision_score(tr_data.Label, tr_pred, average=None,
                                       labels=n_classes)
                p_ts = precision_score(ts_data.Label, ts_pred, average=None,
                                       labels=n_classes)
                r_tr = recall_score(tr_data.Label, tr_pred, average=None,
                                    labels=n_classes)
                r_ts = recall_score(ts_data.Label, ts_pred, average=None,
                                    labels=n_classes)
                a_tr = accuracy_score(tr_data.Label, tr_pred)
                a_ts = accuracy_score(ts_data.Label, ts_pred)
    
                # Make output folder        
                os.makedirs(PATH + folderpath +"/{}_{}".format(lab_type, model),
                            exist_ok=True)
                os.chdir(PATH + folderpath + "/{}_{}".format(lab_type, model))
    
                # Save model
                os.chdir(PATH + folderpath +"/{}_{}".format(lab_type, model))
                featFName = "model_{}_{}.pkl".format(lab_type, model)
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
                feat_data = snv.feat_importance(model, gs, tr_data.columns[:-1])
                fname = "Feature_imp.tsv"
                feat_data.to_csv(fname, header=True, index=False, sep="\t")
                
                # Save predictions
                ## Train
                tr = tr_meta.copy()
                tr["Predicted label"] = tr_pred
                temp = gs.predict_proba(tr_data.drop(["Label"], axis=1))
                for idx, p_class in enumerate(gs.classes_):
                    col_name = "Probability_{}".format(p_class)
                    tr[col_name] = temp[:,idx]
                tr.to_csv("train_predictions.tsv", header=True, index=True,
                          sep="\t")
                ## Test
                ts = ts_meta.copy()
                ts["Predicted label"] = ts_pred
                temp = gs.predict_proba(ts_data.drop(["Label"], axis=1))
                for idx, p_class in enumerate(gs.classes_):
                    col_name = "Probability_{}".format(p_class)
                    ts[col_name] = temp[:,idx]
                ts.to_csv("test_predictions.tsv", header=True, index=True,
                          sep="\t")
    
                # Save metrics
                fname = "metrics_{}_{}.txt".format(lab_type, model)
                with open(fname, 'w') as f:
                    f.write("Data trained on {}\n".format(lab_type))
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
                # If only one label, save only sample statistics
                os.makedirs(PATH + folderpath +"/{}_{}".format(lab_type, model),
                            exist_ok=True)
                os.chdir(PATH + folderpath + "/{}_{}".format(lab_type, model))
                # Save metrics
                fname = "metrics_{}_{}.txt".format(lab_type, model)
                with open(fname, 'w') as f:
                    f.write("Data trained on {}\n".format(lab_type))
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

