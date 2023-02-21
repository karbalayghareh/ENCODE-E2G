import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
import shap
import glob
import os

# Settings for Illustrator:
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['font.family'] = ['Arial','Helvetica']

class E2G:
    '''
    ENCODE-E2G class

    Arguments:

    data_dir: directory containing E-G dataset
    model_dir: directory for saving the models
    fig_dir: directory containing figures
    dataset: CRISPR E-G dataset containing features
    tss: TSS dataset
    feature_table: binary feature assignment table for each model
    extended: whether to train ENCODE-E2G_Extended model
    save_model: save the models as pickle files
    save_predictions: append the E-G prediction sscores to the CRISPR E-G dataset and save it
    epsilon: epsilon value in feature transformation log(|x|+epsilon)
    '''

    def __init__(self, dataset, tss, feature_table, data_dir = './data/crispri', model_dir = './models', fig_dir = './figs', 
                 extended = True, save_model = True, save_predictions = True, epsilon = 0.01):
        
        self.df_dataset = dataset
        self.df_tss = tss
        self.df_feature_table = feature_table
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.fig_dir = fig_dir
        self.extended = extended
        self.save_model = save_model
        self.save_predictions = save_predictions
        self.epsilon = epsilon

        # full model name
        if self.extended:
            self.full_model = 'ENCODE-E2G_Extended'
        else:
            self.full_model = 'ENCODE-E2G'


    def dataset_preprocessing(self):

        '''
        preprocess the CRISPRi E-G dataset
        '''

        self.df_dataset = self.df_dataset[self.df_dataset['measuredGeneSymbol'].isin(self.df_tss['gene'])].reset_index(drop=True)
        self.df_dataset = self.df_dataset[~self.df_dataset['Regulated'].isna()].reset_index(drop=True)
        self.df_dataset = self.df_dataset.replace([np.inf, -np.inf], np.nan)
        self.df_dataset = self.df_dataset.fillna(0)
        df_tss = self.df_tss[self.df_tss['gene'].isin(self.df_dataset['measuredGeneSymbol'])].reset_index(drop=True)
        self.df_dataset['TSS_from_universe'] = -1
        for i, g in enumerate(df_tss['gene'].values):
            idx = self.df_dataset[self.df_dataset['measuredGeneSymbol'] == g].index
            self.df_dataset.loc[idx, 'TSS_from_universe'] = (df_tss.loc[i, 'start'] + df_tss.loc[i, 'end'])//2
        self.df_dataset['distanceToTSS'] = np.abs((self.df_dataset['chromStart'] + self.df_dataset['chromEnd'])//2 - self.df_dataset['TSS_from_universe'])
        self.df_dataset.drop(['TSS_from_universe'], axis = 1, inplace = True)


    def train_and_predict(self):

        '''
        trains logistic regression models, save them as pickle files, and 
        appends the E-G predictions on heldout chromosomes to the dataframe
        with the colum name *.Score
        '''

        # extract the model names
        self.model_list = self.df_feature_table.columns.values[2:]
        print(f'Number of models: {len(self.model_list)}')

        for model_idx, model_name in enumerate(self.model_list):

            # specify feature list
            feature_list = self.df_feature_table[self.df_feature_table[model_name]==1]['features']
            print(f'Model name: {model_name} | Number of features: {len(feature_list)}')

            # transform the features
            X = self.df_dataset.loc[:,feature_list]
            X = np.log(np.abs(X) + self.epsilon)
            Y = self.df_dataset['Regulated'].values.astype(np.int64)

            # logistic regression predictions on chromosome-wise cross validation
            idx = np.arange(len(Y))
            chr_list = np.unique(self.df_dataset['chrom'])
            for chr in chr_list:
                idx_test = self.df_dataset[self.df_dataset['chrom']==chr].index.values
                if model_idx == 0:
                    print(f'Number of E-G pairs in test chromosome {chr} is {len(idx_test)}')

                if len(idx_test) > 0:
                    idx_train = np.delete(idx, idx_test)
                    X_test = X.loc[idx_test, :]
                    X_train = X.loc[idx_train, :]
                    Y_train = Y[idx_train]
                    
                    model = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)

                    if self.save_model:
                        with open(self.model_dir+f'/model_{model_name}_test_{chr}.pkl','wb') as f:
                            pickle.dump(model,f)

                    probs = model.predict_proba(X_test)
                    self.df_dataset.loc[idx_test, model_name+'.Score'] = probs[:,1]

        if self.save_predictions:
            self.df_dataset.to_csv(self.data_dir+f'/{self.full_model}_Predictions.tsv', sep = '\t', index=False)


    def plot_PR_curves(self, save_fig = True):

        '''
        plots precision-recall curves
        '''

        # specify color codes
        if self.extended:
            cmap = matplotlib.cm.get_cmap('tab20', 20)
            color_list = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        else:
            cmap = matplotlib.cm.get_cmap('tab10', 10)
            color_list = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

        for i in range(4):
            if i==0:
                df_dataset_sub = self.df_dataset
            elif i==1:
                df_dataset_sub = self.df_dataset[self.df_dataset['distanceToTSS'] < 10000]
            elif i==2:
                df_dataset_sub = self.df_dataset[(self.df_dataset['distanceToTSS'] >= 10000) & (self.df_dataset['distanceToTSS'] < 100000)]
            elif i==3:
                df_dataset_sub = self.df_dataset[(self.df_dataset['distanceToTSS'] >= 100000) & (self.df_dataset['distanceToTSS'] <= 2500000)]

            Y_true = df_dataset_sub['Regulated'].values.astype(np.int64)
            plt.figure(figsize=(20,20))
            plt.grid()
            c = -1
            for model_name in self.model_list:
                c += 1
                if model_name == 'ABC':
                    Y_pred = df_dataset_sub['ABCScore'].values
                else:
                    Y_pred = df_dataset_sub[model_name+'.Score'].values

                    # number of deleted features
                    n_deleted = len(self.df_feature_table[self.df_feature_table[model_name]==0]['features'])

                precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
                aupr = auc(recall, precision)

                idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
                recall_at_70_pct = recall[idx_recall_70_pct]
                precision_at_70_pct_recall = precision[idx_recall_70_pct]
                threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

                plt.plot(recall, precision, linewidth=5, color=color_list[c], label='{} ({}) || auPR={:6.4f} || Precision={:6.4f} || Threshold={:6.4f}'.format(model_name, n_deleted, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
                if i==0:
                    plt.title('All | #EG = {} | #Positives = {}'.format(len(df_dataset_sub), np.sum(df_dataset_sub['Regulated']==True)), fontsize=40)
                elif i==1:
                    plt.title('[0,10kb) | #EG = {} | #Positives = {}'.format(len(df_dataset_sub), np.sum(df_dataset_sub['Regulated']==True)), fontsize=40)
                elif i==2:
                    plt.title('[10kb,100kb) | #EG = {} | #Positives = {}'.format(len(df_dataset_sub), np.sum(df_dataset_sub['Regulated']==True)), fontsize=40)
                elif i==3:
                    plt.title('[100kb,2.5Mb) | #EG = {} | #Positives = {}'.format(len(df_dataset_sub), np.sum(df_dataset_sub['Regulated']==True)), fontsize=40)

                plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
                plt.xlabel("Recall", fontsize=40)
                plt.ylabel("Precision", fontsize=40)
                plt.tick_params(axis='x', labelsize=40, length=10, width=5)
                plt.tick_params(axis='y', labelsize=40, length=10, width=5)
                plt.grid(False)
                if save_fig:
                    if i==0:
                        plt.savefig(self.fig_dir+f'/{self.full_model}_PR_curve_all.pdf', bbox_inches='tight')
                    elif i==1:
                        plt.savefig(self.fig_dir+f'/{self.full_model}_PR_curve_0-10k.pdf', bbox_inches='tight')
                    elif i==2:
                        plt.savefig(self.fig_dir+f'/{self.full_model}_PR_curve_10k-100k.pdf', bbox_inches='tight')
                    elif i==3:
                        plt.savefig(self.fig_dir+f'/{self.full_model}_PR_curve_100k-2500k.pdf', bbox_inches='tight')


    def model_ablation_bootstrap(self, n_boot = 1000, save_fig = True):
        '''
        compute 95% confidence intervals and p-values of model ablation based on bootstrapping
        '''

        # statistic functions for delta auPR/precision to be used for scipy.stats.bootstrap
        def statistic_delta_aupr(y_true, y_pred_full, y_pred_ablated):
            precision_full, recall_full, thresholds_full = precision_recall_curve(y_true, y_pred_full)
            aupr_full = auc(recall_full, precision_full)

            precision_ablated, recall_ablated, thresholds_ablated = precision_recall_curve(y_true, y_pred_ablated)
            aupr_ablated = auc(recall_ablated, precision_ablated)
            delta_aupr = aupr_ablated - aupr_full

            return delta_aupr

        def statistic_delta_precision(y_true, y_pred_full, y_pred_ablated):
            precision_full, recall_full, thresholds_full = precision_recall_curve(y_true, y_pred_full)
            idx_recall_full_70_pct = np.argsort(np.abs(recall_full - 0.7))[0]
            precision_full_at_70_pct_recall = precision_full[idx_recall_full_70_pct]

            precision_ablated, recall_ablated, thresholds_ablated = precision_recall_curve(y_true, y_pred_ablated)
            idx_recall_ablated_70_pct = np.argsort(np.abs(recall_ablated - 0.7))[0]
            precision_ablated_at_70_pct_recall = precision_ablated[idx_recall_ablated_70_pct]
            delta_precision = precision_ablated_at_70_pct_recall - precision_full_at_70_pct_recall

            return delta_precision

        def bootstrap_pvalue(delta, res_delta):
            '''
            Bootstrap p values for delta (aupr/precision) 
            '''
            
            # Generate boostrap distribution of delta under null hypothesis (important centering step to get sampling distribution under the null)
            delta_boot_distribution = res_delta.bootstrap_distribution - res_delta.bootstrap_distribution.mean()

            # Calculate proportion of bootstrap samples with at least as strong evidence against null    
            pval = np.mean(np.abs(delta_boot_distribution) >= np.abs(delta))

            return pval

        df = pd.DataFrame(columns=['Distance Range', 'Model', 'ID', 'Delta auPR', 'Delta Precision'])
        df_append = pd.DataFrame(columns=['Distance Range', 'Model', 'ID', 'Delta auPR', 'Delta Precision'])
        for i in range(4):
            if i==0:
                df_dataset_sub = self.df_dataset
                Distance_Range = 'All'
            elif i==1:
                df_dataset_sub = self.df_dataset[self.df_dataset['distanceToTSS'] < 10000]
                Distance_Range = '[0, 10kb)'
            elif i==2:
                df_dataset_sub = self.df_dataset[(self.df_dataset['distanceToTSS'] >= 10000) & (self.df_dataset['distanceToTSS'] < 100000)]
                Distance_Range = '[10kb, 100kb)'
            elif i==3:
                df_dataset_sub = self.df_dataset[(self.df_dataset['distanceToTSS'] >= 100000) & (self.df_dataset['distanceToTSS'] <= 2500000)]
                Distance_Range = '[100kb, 2.5Mb)'

            ## scipy bootstrap
            Y_true = df_dataset_sub['Regulated'].values.astype(np.int64)
            Y_pred_full = df_dataset_sub[self.model_list[0]+'.Score'].values
            for model_name in self.model_list[1:]:
                print(f'Distance range: {Distance_Range}, model: {model_name}')

                # number of deleted features
                n_deleted = len(self.df_feature_table[self.df_feature_table[model_name]==0]['features'])
                model_name_n_deleted = model_name+ f' ({n_deleted})'

                Y_pred_ablated = df_dataset_sub[model_name+'.Score'].values
                data = (Y_true, Y_pred_full, Y_pred_ablated)
                delta_aupr = statistic_delta_aupr(Y_true, Y_pred_full, Y_pred_ablated)
                delta_precision = statistic_delta_precision(Y_true, Y_pred_full, Y_pred_ablated)

                res_delta_aupr = scipy.stats.bootstrap(data, statistic_delta_aupr, n_resamples=n_boot, paired=True, confidence_level=0.95, method='percentile')
                res_delta_precision = scipy.stats.bootstrap(data, statistic_delta_precision, n_resamples=n_boot, paired=True, confidence_level=0.95, method='percentile')

                print(f'Delta auPR p-value = {bootstrap_pvalue(delta_aupr, res_delta_aupr)}')
                print(f'Delta precision p-value = {bootstrap_pvalue(delta_precision, res_delta_precision)}')
                print('###################################################################################')

                df_append['Delta auPR'] = res_delta_aupr.bootstrap_distribution
                df_append['Delta Precision'] = res_delta_precision.bootstrap_distribution
                df_append['ID'] = np.arange(n_boot)
                df_append['Distance Range'] = Distance_Range
                df_append['Model'] = model_name_n_deleted
                df = pd.concat([df, df_append], ignore_index=True)

        df_sub = df[df['Distance Range']=='All']
        median_dict = {}
        model_list_ablated = np.unique(df['Model'])
        for model_name in model_list_ablated:
            if model_name != "ENCODE-E2G_Extended (0)":
                median_dict[model_name] = np.mean(df_sub[df_sub['Model']==model_name]['Delta auPR'].values)

        sorted_list = sorted([(value,key) for (key,value) in median_dict.items()])
        order = []
        for _, model in enumerate(sorted_list):
            order.append(model[1])

        if self.extended:
            palette = 'tab20'
        else:
            palette = 'tab10'
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25, 6))
        ax1.grid('y')
        g = sns.barplot(data=df, x='Distance Range', y='Delta auPR', hue='Model', ax=ax1, errorbar=("pi", 95), seed=None, hue_order=order, errwidth=1.5, capsize=0.03, palette=palette)
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
        ax1.yaxis.set_tick_params(labelsize=20)
        ax1.xaxis.set_tick_params(labelsize=20)
        g.set_xlabel("",fontsize=20)
        g.set_ylabel("Delta auPR",fontsize=20)
        plt.setp(ax1.get_legend().get_texts(), fontsize='20')
        plt.setp(ax1.get_legend().get_title(), fontsize='20')
        plt.tight_layout()
        if save_fig:
            plt.savefig(self.fig_dir+f'/{self.full_model}_Model_Ablation_Barplot_Delta_auPR.pdf', bbox_inches='tight')

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25, 6))
        ax1.grid('y')
        g = sns.barplot(data=df, x='Distance Range', y='Delta Precision', hue='Model', ax=ax1, errorbar=("pi", 95), seed=None, hue_order=order, errwidth=1.5, capsize=0.03, palette=palette)
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
        ax1.yaxis.set_tick_params(labelsize=20)
        ax1.xaxis.set_tick_params(labelsize=20)
        g.set_xlabel("",fontsize=20)
        g.set_ylabel("Delta Precision",fontsize=20)
        plt.setp(ax1.get_legend().get_texts(), fontsize='20')
        plt.setp(ax1.get_legend().get_title(), fontsize='20')
        plt.tight_layout()
        if save_fig:
            plt.savefig(self.fig_dir+f'/{self.full_model}_Model_Ablation_Barplot_Delta_Precision.pdf', bbox_inches='tight')
            

    def compute_shap(self, plot_shap = True):
        '''
        compute SHAP scores of full models and plot them
        '''

        # get feature list
        feature_list = self.df_feature_table['features']
        print(f'Model: {self.full_model} | Number of features: {len(feature_list)}')

        # transform the features
        X = self.df_dataset.loc[:,feature_list]
        X = np.log(np.abs(X) + self.epsilon)
        Y = self.df_dataset['Regulated'].values.astype(np.int64)

        # logistic regression predictions on chromosome-wise cross validation
        idx = np.arange(len(Y))
        shap_values_all = np.empty([0,X.shape[1]])
        X_test_all = pd.DataFrame(columns=feature_list)
        chr_list = np.unique(self.df_dataset['chrom'])
        for chr in chr_list:
            idx_test = self.df_dataset[self.df_dataset['chrom']==chr].index.values
            print(f'Number of E-G pairs in test chromosome {chr} is {len(idx_test)}')

            if len(idx_test) > 0:
                idx_train = np.delete(idx, idx_test)
                X_test = X.loc[idx_test, :]
                X_train = X.loc[idx_train, :]
                Y_train = Y[idx_train]
                
                # model
                model = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train.values, Y_train)

                # compute shap scores
                background = shap.kmeans(X_train, 10)  # use 10 kmeans samples from train samples as background
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_test)
                shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
                X_test_all = pd.concat([X_test_all, X_test], ignore_index=True)

        if plot_shap:
            fig=plt.gcf()
            shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
            fig.savefig(self.fig_dir+f'/{self.full_model}_SHAP_scores_barplot.pdf', bbox_inches='tight')

            fig=plt.gcf()
            shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
            fig.savefig(self.fig_dir+f'/{self.full_model}_SHAP_scores_dotplot.pdf', bbox_inches='tight')


    def predict_genomewide(self, append_predictions=False):
        '''
        predict E-G scores genomewide for all the genes using 
        ENCODE-E2G models trained on CRISPRi E-G dataset
        '''

        if self.extended:
            genomewide_predictions_dir = './data/genomewide_predictions/encode_e2g_extended'
        else:
            genomewide_predictions_dir = './data/genomewide_predictions/encode_e2g'

        # specify feature list
        feature_list = self.df_feature_table[self.df_feature_table[self.full_model]==1]['features']
        feature_list_in_file = feature_list+'.Feature'
        print(f'Model name: {self.full_model} | Number of features: {len(feature_list)}')

        k = 0
        for filepath in glob.glob(os.path.join(genomewide_predictions_dir, '*.gz')):
            if 'Predictions' not in filepath:

                k += 1
                print(f'File number = {k} | filepath = {filepath}')

                df_enhancers = pd.read_csv(filepath, delimiter = '\t')
                df_enhancers = df_enhancers.replace([np.inf, -np.inf], np.nan)
                df_enhancers = df_enhancers.fillna(0)

                # transform the features
                X = df_enhancers.loc[:,feature_list_in_file]
                X.columns = feature_list
                X = np.log(np.abs(X) + self.epsilon)
                chr_list = np.unique(self.df_dataset['chrom'])

                for chr in chr_list:
                    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values

                    if len(idx_test) > 0:
                        X_test = X.loc[idx_test, :]

                        with open(self.model_dir+f'/model_{self.full_model}_test_{chr}.pkl','rb') as f:
                            model = pickle.load(f)

                        probs = model.predict_proba(X_test)
                        df_enhancers.loc[idx_test, self.full_model+'.Score'] = probs[:,1]

                if append_predictions:
                    df_enhancers.to_csv(filepath, sep = '\t', index=False)
                else:
                    df_enhancers.to_csv(filepath.replace('.tsv.gz', '_Predictions.tsv.gz'), sep = '\t', index=False)
