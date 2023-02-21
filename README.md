# ENCODE-E2G

## Install requirements

`conda create --name <env> --file requirements.txt`

## Run 

You need three files to run ENCODE-E2G models:

    - TSS file
    - CRISPRi E-G (enhancer-gene) dataset
    - Feature table

[tss](https://github.com/karbalayghareh/ENCODE-E2G/tree/main/tss) contains the TSS file we have used. [crispri](https://github.com/karbalayghareh/ENCODE-E2G/tree/main/data/crispri) has the CRISPRi datasets for training the ENCODE-E2G and ENCODE-E2G_Extended models. Feature tables are binary matrices which specify the features to be used in each model. We have included the the full and ablated models of ENCODE-E2G and ENCODE-E2G_Extended in [feature_table](https://github.com/karbalayghareh/ENCODE-E2G/tree/main/data/feature_table).

This [demo](https://github.com/karbalayghareh/ENCODE-E2G/blob/main/demo.ipynb) file shows step-by-step how to run the ENCODE-E2G models, save them, predict the CRISPRi E-G pair, plot the analysis figures, and perform the genome-wide predictions for the provided E-G pairs for every ENCODE cell types. 