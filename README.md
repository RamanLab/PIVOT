# PIVOT
PIVOT (<ins>P</ins>ersonalised <ins>I</ins>dentification of dri<ins>V</ins>er <ins>O</ins>ncogenes and <ins>T</ins>umour suppressors) is a tool used to identify personalised tumour suppressor genes (TSGs) and oncogenes (OGs) using multi-omic data. The genes are labelled at for each patient as TSG or OG.

## Table of Contents

- [Description](#description)
- [Data](#data)
- [Folder structure](#folder-structure)
- [Installation](#installation)
- [Using PIVOT](#using-pivot)
- [Links](#links)

## Description

PIVOT is the first supervised tool to predict personalised driver genes and label them based on functionality as TSG or OG. The best model is trained using multi-omic features using labels from Bailey *et al.* Our model predicts well known driver genes as well as new driver genes that are frequently altered across samples. The predictions are made for genes mutated or altered by copy number variations. PIVOT predicts labels on individual samples and can be used on as few as a single patient. The features are independent of other samples and can be hence used in a clinical setting. PIVOT identifies rare drivers altered in as few as one sample.

While our best model is dependent on multi-omic features, we also publish SNV and RNA feature based models that can be used on SNV and RNA features respectively. 

## Data
The TCGA data for BRCA, COAD, LGG and LUAD was downloaded from GDC. 
The preprocessed data can be found below:
- [BRCA](https://doi.org/10.5281/zenodo.5898117)
- [COAD](https://doi.org/10.5281/zenodo.5898163)
- [LGG](https://doi.org/10.5281/zenodo.5898031)
- [LUAD](https://doi.org/10.5281/zenodo.5898190)

## Folder structure
The top-level directories contain code, data and output folders. 

### The directory layout

    .
    ├── code                            # All the code for analysis
    ├── data
    │   ├── domains                     # pre-processed domain files required for feature generation
    │   │   └── pfam                    
    │   ├── driver genes                # List of driver genes for labels
    │   │   ├── Bailey et al
    │   │   ├── CGC
    │   │   ├── CIViC
    │   │   └── Martelotto et al
    │   ├── miRNA                       # pre-processed miRNA files required for feature generation
    │   ├── network                     # pre-processed network files required for feature generation
    │   └── neutral.txt                 # List of neutral genes for labels
    ├── output
    │   ├── GDC_BRCA                    # Results for cancer-type BRCA
    │   │   ├── multiomic               # Metrics, plots for all models built using multi-omic features
    │   │   ├── predict                 # Predictions using PIVOT
    │   │   ├── RNA                     # Metrics, plots for all models built using RNA features
    │   │   └── SNV                     # Metrics, plots for all models built using SNV features
    │   ├── GDC_COAD                    # Results for cancer-type COAD
    │   │   └── ...
    │   ├── GDC_LGG            	        # Results for cancer-type LGG
    │   │   └── ...
    │   └── GDC_LUAD                    # Results for cancer-type LUAD
    │       └── ...
    └── README.md

The code folder containes all the files used for building the feature matrix, building the models and and the tissue-specific analysis.

    .
    ├── ...
    ├── code                                # All the code for analysis
    │   ├── analyse_predictions_BRCA.ipynb  # Analyse and plot data from predictions of BRCA
    │   ├── analyse_predictions_COAD.ipynb  # Analyse and plot data from predictions of COAD
    │   ├── analyse_predictions_LUAD.ipynb  # Analyse and plot data from predictions of LUAD
    │   ├── PIVOT_predict.ipynb             # Notebook to generate features and predict labels
    │   ├── multiomic_classifier.py         # Generates all multi-omic models
    │   ├── preprocess_CNV.ipynb            # Pre-process CNV data
    │   ├── preprocess_drivers.ipynb        # Pre-process driver lists
    │   ├── preprocess_miRNA.ipynb          # Pre-process miRNA data
    │   ├── preprocess_networks.ipynb       # Pre-process network data
    │   ├── preprocess_SNV.ipynb            # Pre-process SNV data
    │   ├── preprocessRNA.Rmd               # Pre-process RNA data
    │   ├── rna_classifier.py               # Generates all RNA models
    │   ├── multiomic_classifier.py         # Generates all multi-omic models
    │   └── snp_classifier.py               # Generates all SNV models
    └── ...

## Installation
Download PIVOT from GitHub and add the folder to PYTHONPATH.

PIVOT requires the following dependencies to run smoothly:
- Python >3
- numpy 1.20.3
- pandas 1.3.4
- sklearn 0.24.2
- imblearn 0.8.0

## Using PIVOT
- Download [COAD data](https://doi.org/10.5281/zenodo.5898163).
- Open `PIVOT_predict.ipynb` notebook and follow thw steps


## Links
[BioRxiv paper](https://doi.org/10.1101/2022.01.13.476163)
