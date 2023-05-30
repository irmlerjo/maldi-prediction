import pandas as pd
import numpy as np
import imblearn
import random
from collections import Counter
from os.path import exists
import logging

import maldi_learn.utilities as ml_utilities
import maldi_learn.driams as ml_driams
import maldi_learn.filters as ml_filters

import sys 
del sys.modules['const']
import const


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

from sklearn import tree
from sklearn import dummy
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import optuna
import math

import pickle 



class OptimizerModel:    
    def __init__(self,dataset_label,method, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.dataset_label=dataset_label
        self.method = method


    # Best Params is Dict with best params learned after optimizing
    def set_best_params(self, best_params):
        self.best_params=best_params

    # Predicted Labels and Probas are two sets which contain the predicted labels for the test data
    def set_predicted_labels(self,predicted_labels, predicted_probas):
        self.predicted_labels = predicted_labels
        self.predicted_probas = predicted_probas




class Utility:


######################################################## Load Data ################################################################
        
    ## Load Data from Driams
    # - load_data(...)
    #   - read in data from DRIAMS DB
    #       - Params are: 
    #       - bacterial_species: bacterial species that should be filtered by (other labels are not loaded)
    #       - predicted_antibiotic: antibiotic of which the resistance/susceptibility is to be predicted by ML (everything that has no labels S/R for the antibiotic is filtered out)
    #       - driams (one of DRIAMS_A to DRIAMS_D)
    #       - years (either one or an array (only other than 2018 for DRIAMS_A))
    ## Loading Data
    #   - which data is being loaded
    #   - load the data and split it into train/test dataset.
    def load_data(bacterial_species,predicted_antibiotic,driams,years,binning):
        # which binning to use?
        prebinned=None
        if(binning==const.BINNING_6K):
            prebinned=True
        elif(binning==const.BINNING_18K):
            prebinned=False
            

        spectra_type = 'binned_6000'
        if(prebinned==False):
            spectra_type = 'preprocessed'
        elif(binning==const.BINNING_18K_RAW):
            spectra_type = 'raw'

        # folder where the data is being loaded from
        ml_driams.DRIAMS_ROOT = "F:\\extracted_driams\\"


        extra_filters=[]
        # TODO: Hospital Hygien Filter What does it do, what can i do?
        if(driams=='DRIAMS_A'):
            extra_filters.append(
                    ml_filters.DRIAMSBooleanExpressionFilter('workstation != HospitalHygiene')
            )
            suffix='strat'
            # Important as in Driams A are multiple spectra from the same patient!!!!
            strat_fn = ml_utilities.case_based_stratification
        else:
            suffix='clean'
            strat_fn = ml_utilities.stratify_by_species_and_label

        driams_dataset = ml_driams.load_driams_dataset(
            ml_driams.DRIAMS_ROOT,
            driams,
            years,
            species=bacterial_species,
            antibiotics=predicted_antibiotic,
            handle_missing_resistance_measurements='remove_if_all_missing',
            spectra_type=spectra_type,
            on_error='warn',
            id_suffix=suffix, # TODO: When clean and when strat?
            extra_filters=extra_filters,
        )
        # Used for RAW or PREPROCESSED
        if((binning==const.BINNING_18K_RAW)|(prebinned==False)):
            Utility.bin_dataset(driams_dataset)

        train_indices = {}
        test_indices={}
        y = {}

        # Stratified Split and Bin Data
        for antibiotic in predicted_antibiotic:
            # TODO: REMOVE STUB STUFF
            train_index,test_index = strat_fn(driams_dataset.y,
            antibiotic=antibiotic,
            random_state=42)

            train_indices[antibiotic] = train_index
            test_indices[antibiotic] = test_index
            y[antibiotic] = driams_dataset.to_numpy(antibiotic)
            #meta = driams_dataset.y[antibiotic].drop(columns=antibiotic)


        if((prebinned==False)|(binning==const.BINNING_18K_RAW)):   # Used for self Binned data of 1Da            
            X = np.asarray([spectrum for spectrum in driams_dataset.X])
        elif(prebinned==True):                   # Used for binned_6000 data            
            X = np.asarray([spectrum.intensities for spectrum in driams_dataset.X])
        else:
            raise ValueError(binning)

        X_train = {}
        y_train = {}
        X_test = {}
        y_test = {}
        for index in train_indices:
            X_train[index] = X[train_indices[index]]
            y_train[index] = y[index][train_indices[index]]
        for index in test_indices:
            X_test[index] = X[test_indices[index]]
            y_test[index] = y[index][test_indices[index]]
        #meta_train, meta_test = meta.iloc[train_index], meta.iloc[test_index]
        return X_train, y_train, X_test, y_test,driams_dataset

    # Bin spectra in buckets
    def bin_spectrum(spectrum,method='sum'):
        bins = []
        index=0
        for i in range(2000,20000):
            bucket=0
            while (index < len(spectrum)) and (spectrum[index][0] < i):
                bucket=bucket+spectrum[index][1]
                index=index+1
            bins.append(bucket)
        return bins

    def bin_dataset(dataset):
        for i in range(0,dataset.n_samples):
            dataset.X[i] = Utility.bin_spectrum(dataset.X[i])


######################################################## Undersampling Dataset ################################################################


    # Undersample the dataset with the given Type (US_IMBLEARN, US_RANDOM or US_NO)
    # Return the undersampled dataset
    # - One method to handle imbalanced data
    # - method delete_random: delete random labels to handle imbalanced data 
    # - TODO: Welche Möglichkeiten gibt es noch außer Undersampling?
    def select_undersampling_dataset(X,y,type):
        if(type== const.US_IMBLEARN):
            # Train Resample
            cnn = imblearn.under_sampling.CondensedNearestNeighbour(random_state=42)
            X,y = cnn.fit_resample(X,y)
            return X,y
        elif(type==const.US_RANDOM):
            random.seed(42)
            while Counter(y).get(0) > Counter(y).get(1):
                try_int = random.randrange(y.size)
                if y[try_int]==0:
                    X = np.delete(X, try_int, 0)
                    y = np.delete(y, try_int, 0)
            return X,y
        elif(type==const.US_NO):
            return X,y
        else:
            raise Exception("Wrong Value for Undersampling")

    
################################################### MODEL SELECTION ######################################################################################

################################# Predict with actual data ####################################################################

    # ## Method predict_labels
    #    - params: 
    #        - Classifier (tree, dummy, etc.)
    #        - X_train, y_train, X_test
    # For Baseline Methods
    def predict_labels(classifier,X_train,y_train,X_test):
        classifier.fit(X_train,y_train)
        y_test_pred_proba = classifier.predict_proba(X_test)
        y_test_pred = classifier.predict(X_test)    
        return y_test_pred, y_test_pred_proba

    # Predict Labels for Tree
    def predict_with_best_tree(X_train,y_train,X_test,best_params):
        clf = tree.DecisionTreeClassifier(
        criterion = best_params.get('criterion'),
        class_weight = best_params.get('class_weight'),
        random_state=const.RANDOM_STATE) # Change Random State for trying to train? : Use
        return Utility.predict_labels(clf,X_train,y_train,X_test)

    # Predict Labels for Dummy
    def predict_with_best_dummy(X_train,y_train,X_test,best_params):
        clf = dummy.DummyClassifier(strategy=best_params.get('strategy')) # Change Random State for trying to train? : Use
        return Utility.predict_labels(clf,X_train,y_train,X_test)

    # Predict Labels for Logistic Regression
    def predict_with_best_lr(X_train,y_train,X_test,best_params):
            clf = linear_model.LogisticRegression(
                            solver='saga',
                            max_iter=500,
                            class_weight='balanced',
                            random_state=const.RANDOM_STATE,
                            C=best_params.get('C'),
                            penalty=best_params.get('penalty')
                    )
            return Utility.predict_labels(clf,X_train,y_train,X_test)

    # Predict Labels for Random Forest
    def predict_with_best_rfo(X_train,y_train,X_test,best_params):        
        clf = RandomForestClassifier(
            class_weight='balanced',
            n_jobs=-1,
            random_state=const.RANDOM_STATE,
            criterion=best_params.get('criterion'),
            n_estimators=best_params.get('n_estimators'),
            max_features=best_params.get('max_features'))
        return Utility.predict_labels(clf,X_train,y_train,X_test)

    # Predict Labels for actual Test Data
    def predict_with_best(optimizer_model,X_test):
        if(optimizer_model.method==const.METHOD_DUMMY):
            return Utility.predict_with_best_dummy(optimizer_model.X_train,optimizer_model.y_train,X_test,optimizer_model.best_params)
        elif(optimizer_model.method==const.METHOD_TREE):
            return Utility.predict_with_best_tree(optimizer_model.X_train,optimizer_model.y_train,X_test,optimizer_model.best_params)
        elif(optimizer_model.method==const.METHOD_RFO):
            return Utility.predict_with_best_rfo(optimizer_model.X_train,optimizer_model.y_train,X_test,optimizer_model.best_params)
        elif(optimizer_model.method==const.METHOD_LR):
            return Utility.predict_with_best_lr(optimizer_model.X_train,optimizer_model.y_train,X_test,optimizer_model.best_params)
        else:
            raise Exception
       

################################# Optimize Baseline ####################################################################

    # Search for optimal params, using only Train Data.
    # This is for baseline methods which use GridsearchCV for optimal search    
    def perform_gridsearch(classifier,X_train,y_train,param_grid):
        gridsearch = GridSearchCV(classifier,param_grid,scoring='roc_auc',cv=5) # 5-Fold Cross validation is also default in Gridsearch
        gridsearch.fit(X_train,y_train)  
        return gridsearch.best_params_

    # optimize Tree
    def optimize_tree(X_train,y_train):
        param_grid = const.PARAM_GRID_TREE
        clf = tree.DecisionTreeClassifier(random_state=const.RANDOM_STATE)
        return Utility.perform_gridsearch(clf,X_train,y_train,param_grid)

    # Optimize Random Forest
    def optimize_rfo(X_train,y_train):
        param_grid = const.PARAM_GRID_RFO
        clf = RandomForestClassifier(
            n_jobs=-1,
            random_state=const.RANDOM_STATE)
        return Utility.perform_gridsearch(clf,X_train,y_train,param_grid)

    # Optimize Dummy
    def optimize_dummy(X_train,y_train):
        param_grid = const.PARAM_GRID_DUMMY
        clf = dummy.DummyClassifier()
        return Utility.perform_gridsearch(clf,X_train,y_train,param_grid)

    # Optimize LogReg
    def optimize_lr(X_train,y_train):        
        param_grid = const.PARAM_GRID_LR
        clf = linear_model.LogisticRegression(solver='saga',
                            max_iter=500,
                            random_state=const.RANDOM_STATE)
        return Utility.perform_gridsearch(clf,X_train,y_train,param_grid)

    # Returns best Params
    def optimize(optimizer_model):        
        if(optimizer_model.method==const.METHOD_DUMMY):
            return Utility.optimize_dummy(optimizer_model.X_train,optimizer_model.y_train)
        elif(optimizer_model.method==const.METHOD_TREE):
            return Utility.optimize_tree(optimizer_model.X_train,optimizer_model.y_train)
        elif(optimizer_model.method==const.METHOD_RFO):
            return Utility.optimize_rfo(optimizer_model.X_train,optimizer_model.y_train)
        elif(optimizer_model.method==const.METHOD_LR):
            return Utility.optimize_lr(optimizer_model.X_train,optimizer_model.y_train)
        else:
            raise Exception


    ################################################# Optimize Deep Learning #######################################################

    # Train for given number of epochs. Use training_iterator for training and validation_iterator for Validation.
    def training_epochs(epochs,training_iterator,validation_iterator,model,optimizer,criterion,device,weighted_flag):

        scores_test = []
        scores_training = []

        preds = []
        probas = []

        for epoch in range(epochs):

            # Optimize only on the Training data
            training_proba,training_actual_y,train_loss = Utility.train(model, training_iterator, optimizer, criterion, device, weighted_flag)

            #Evaluate with Validation Data
            with torch.no_grad():
                loss, pred, proba, actual_y = Utility.evaluate(model, validation_iterator, criterion, device)
            # Add all Batch results to one list

            # Evaluation Stuff
            pred = torch.cat(pred, dim=0)
            proba = torch.cat(proba, dim=0)
            preds.append(pred)
            probas.append(proba)
            actual_y = torch.cat(actual_y, dim=0)
            proba_cpu = proba.cpu().detach().numpy()
            actual_y_cpu = actual_y.cpu().detach().numpy()
            aucroc_test=metrics.roc_auc_score(actual_y_cpu,proba_cpu)
            # Use Combined Auroc and AUPRC?
            #auprc_test = metrics.average_precision_score(actual_y_cpu, proba_cpu)
            #combined_score=aucroc_test*auprc_test
            scores_test.append(aucroc_test)
            

            # Training Stuff
            training_proba = torch.cat(training_proba, dim=0)
            training_actual_y = torch.cat(training_actual_y, dim=0)
            training_proba_cpu = training_proba.cpu().detach().numpy()
            training_actual_y_cpu = training_actual_y.cpu().detach().numpy()
            aucroc_training=metrics.roc_auc_score(training_actual_y_cpu,training_proba_cpu)
            scores_training.append(aucroc_training)
            #writer.add_scalars('CNN_Loss',{'Loss_Training': train_loss,'Loss_Validation': loss},epoch)
            #writer.add_scalars('CNN_AUROC',{'AUC_Training': aucroc_training,'AUC_Validation': aucroc_test},epoch)



            #print(metrics.roc_auc_score(actual_y_cpu,proba_cpu)) Before it was Loss!
        return aucroc_test, pred, proba, scores_test, preds, probas,scores_training


    # Train
    def train(model, iterator, optimizer, criterion, device,weighted_flag):
        if(weighted_flag):
            pos_class_ratio = Utility.calculate_pos_class_ratio(iterator)
            class_add = torch.tensor([pos_class_ratio]).cuda() # ~ 0,1 for S-OXA
            class_weights = torch.tensor([1-pos_class_ratio]).cuda() # ~0,9 for S-OXA

        model.train()
        proba = []
        actual_y = [] 

        for (x, y) in iterator:
            # Reset Optimizer Gradient
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)


            # this calls forward
            #writer.add_graph(model,x)
            #writer.flush()
            y_pred = model(x)
            if(const.OUTPUT_DIM==1):
                #y_pred = torch.sigmoid(y_pred)
                y_pred = torch.flatten(y_pred)
                y_pred = y_pred.to(torch.float64)
                y = y.to(torch.float64)
                proba.append(y_pred)
                actual_y.append(y)
            if(weighted_flag):
                batch_weight= (y*class_weights)+class_add
                curr_criterion = nn.BCELoss(weight=batch_weight) # TODO: add weight as trial parameter! weight=batch_weight)
                curr_criterion.to(device)
            else:
                curr_criterion=criterion
            loss = curr_criterion(y_pred, y)
            #loss_normal = criterion(y_pred, y)
            #logging.warning("loss_a",loss)
            #logging.warning("loss_reg",loss_normal)
            loss.backward()            

            optimizer.step()
        return proba,actual_y,loss

    # Evaluate Model with iterator 
    # Return added loss from all items
    def evaluate(model, iterator, criterion, device):
        epoch_loss = 0  
        pred = []
        proba = []  
        actual_y = [] 

        model.eval()

        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            
            if(const.OUTPUT_DIM==1):                
                y_pred = torch.flatten(y_pred)
                y_pred = y_pred.to(torch.float64)
                y = y.to(torch.float64)  

            loss = criterion(y_pred, y)
            epoch_loss += loss.item()

            if(const.OUTPUT_DIM==1):
                curr_proba=y_pred
                pred.append(torch.round(curr_proba))
                #pred=torch.le(curr_proba, 0.5).int()
                proba.append(curr_proba)
                actual_y.append(y)




            # Append Predictions
            if(const.OUTPUT_DIM==2):
                #curr_proba = torch.softmax(y_pred,1)
                pred.append(torch.argmax(curr_proba, 1))
                proba.append(curr_proba)

        return epoch_loss / len(iterator), pred, proba, actual_y
    
    # Train model one time with training data and evaluate test data. Returns predictions and prediction probabilities.
    def predict_with_best_py(best_params,train_iterator,test_iterator,method,n_bins):

        # Initialize Model with given Method and Parameters
        model,optimizer,criterion,device = Utility.initialize_model(best_params,method,n_bins)        

        # Train over given number of Epochs, with best Parameters
        aucroc, pred, proba, scores,preds,probas,training_scores = Utility.training_epochs(best_params[const.EPOCHS],train_iterator,test_iterator,model,optimizer,criterion,device)

        # Return Predictions and Probabilities
        return pred, proba, scores,preds,probas,training_scores

        


#################################################### EVALUATION ########################################################################################

    ## Method plot_confusion_matrix
    #    - method for plotting a confusion matrix
    #    - params are labels and predicted labels, and a headline which shows the method.
    def plot_confusion_matrix(labels, pred_labels,method,normalize,bacterial_species,predicted_antibiotic):
        fig = plt.figure(figsize=(10, 10))
        cm = confusion_matrix(labels, pred_labels,normalize=normalize) 
        cm = ConfusionMatrixDisplay(cm, display_labels=['S','R'])
        cm.plot(cmap='Blues')
        plt.title(method)
        if(normalize=='true'):
            plt.savefig('results/'+bacterial_species+predicted_antibiotic+method+'_normalized.png')
        else:
            plt.savefig('results/'+bacterial_species+predicted_antibiotic+method+'.png')
    
    # plot AUROC, probas contains dict indexed by  label
    def plot_auc_roc(y_test,y_probas,dataset_label,bacterial_species,predicted_antibiotic):
        plt.figure(dataset_label).clf()
        plt.title(dataset_label)
        for i in y_probas:
            if(const.OUTPUT_DIM==1):
                fpr, tpr, thresh = metrics.roc_curve(y_test, y_probas[i])
                auc = metrics.roc_auc_score(y_test, y_probas[i])
            else:
                fpr, tpr, thresh = metrics.roc_curve(y_test, y_probas[i][:,1])
                auc = metrics.roc_auc_score(y_test, y_probas[i][:,1])
            plt.plot(fpr,tpr,label=f"{i}, AUCROC={auc:.2f}")


        plt.ylabel("True-positive Rate")
        plt.xlabel("False-positive Rate")

        plt.legend(loc=0)

        plt.savefig('results/'+bacterial_species+'_'+predicted_antibiotic+'_'+dataset_label+'_aucroc.png')
                    
            
    # plot AUPRC, probas contains dict indexed by label      
    def plot_auc_prc(y_test,y_probas,dataset_label,bacterial_species,predicted_antibiotic):
        plt.figure(dataset_label).clf()
        plt.title(dataset_label)
        for i in y_probas:
            if(i==const.METHOD_DUMMY):
                prc = metrics.average_precision_score(y_test, y_probas[i])
                plt.plot([],label=f"{i}, (AP={prc:.2f})")
            else:
                if(const.OUTPUT_DIM==1):
                    prec, reca, thresh = metrics.precision_recall_curve(y_test, y_probas[i])
                    prc = metrics.average_precision_score(y_test, y_probas[i])
                else:
                    prec, reca, thresh = metrics.precision_recall_curve(y_test, y_probas[i][:,1])
                    prc = metrics.average_precision_score(y_test, y_probas[i][:,1])
                plt.plot(reca,prec,label=f"{i}, (AP={prc:.2f})")


        plt.ylabel("Precision")
        plt.xlabel("Recall")

        plt.legend(loc=0)

        plt.savefig('results/'+bacterial_species+'_'+predicted_antibiotic+'_'+dataset_label+'_aucprc.png')
    
    # Load or save Stuff
    # Probas
    def save_probas(filename,probas):
        np.savetxt('results/probas/'+filename+'_'+'probas'+'.csv',probas,delimiter=';')
    def load_probas(filename):
        return np.loadtxt('results/probas/'+filename+'_'+'probas'+'.csv',delimiter=';',dtype=np.dtype('>f8'))
    # Predictions
    def save_predictions(filename,preds):
        np.savetxt('results/predictions/'+filename+'_'+'preds'+'.csv',preds,delimiter=';')
    def load_predictions(filename):
        return np.loadtxt('results/predictions/'+filename+'_'+'preds'+'.csv',delimiter=';',dtype=np.dtype('>i4'))
    # Losses
    def save_losses(filename,losses):
        np.savetxt('results/losses/'+filename+'_'+'losses'+'.csv',losses,delimiter=';')
    def load_losses(filename):
        return np.loadtxt('results/losses/'+filename+'_'+'losses'+'.csv',delimiter=';')
    # Best Params
    def save_best_params(filename,best_params):
        with open('results/params/'+filename+'_'+'best_params'+'.pkl', 'wb') as f:
            pickle.dump(best_params, f)      
    def load_best_params(filename):
        with open('results/params/'+filename+'_'+'best_params'+'.pkl', 'rb') as f:
            return pickle.load(f)
    
    
    # Filename contains: Antibiotic, Bacterial Strain, Method, Undersampling Method, Binning, Dataset
    def create_file_identifier(antibiotic,bacteria,method,undersampling,binning,dataset):
        return antibiotic+'_'+bacteria+'_'+undersampling+'_'+binning+'_'+dataset+'_'+method+'_'+'aucroc_eval'



#################################################### Helper Methods ############################################################################################
    
    def create_iterator(X_tensor,y_tensor,shuffle,batch_size):
        X_tensor = Utility.unsqueeze(X_tensor)
        dataset = data.TensorDataset(X_tensor,y_tensor)
        return data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    
    def calculate_pos_class_ratio(iterator):
        counter=0
        counterpos=0
        for (batch_x,batch_y) in iterator:
            for single_y in batch_y:
                counter=counter+1
                if(single_y==1):
                    counterpos=counterpos+1
        return counterpos/counter


    # Split Train data into Training and Validation Data
    # Returns all the splits
    def split_dataset(X_tensor,y_tensor, stratify,batch_size):    
        X_tensor = Utility.unsqueeze(X_tensor)   
        if(stratify):
            fold = StratifiedKFold(n_splits=5,shuffle=True,random_state=const.RANDOM_STATE)
        else:
            fold = KFold(n_splits=5,shuffle=True,random_state=const.RANDOM_STATE)
        train_data = {}
        valid_data = {}
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(X_tensor, y_tensor)):
            train_X_tensor = X_tensor[train_idx]
            valid_X_tensor = X_tensor[valid_idx]
            train_y_tensor = y_tensor[train_idx]
            valid_y_tensor = y_tensor[valid_idx]

            train_dataset = data.TensorDataset(train_X_tensor,train_y_tensor)
            valid_dataset = data.TensorDataset(valid_X_tensor,valid_y_tensor)
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            valid_loader = data.DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            train_data[fold_idx] = train_loader
            valid_data[fold_idx] = valid_loader

        return train_data,valid_data

    # Add Channel Dimension and create dataset
    def unsqueeze(X):
        unsqueezed_tensor = X.unsqueeze(1)
        return unsqueezed_tensor

    def cluster_pca(X_train,y_train):
        k_cluster = KMeans(n_clusters=2,max_iter=500,random_state=200,algorithm="lloyd")
        k_cluster.fit(X_train)
        arr=k_cluster.fit_predict(X_train)
        np.count_nonzero(arr==1) #

        # Just PCA, without taking into account the labels.


        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()

        ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
        ax.set_position([0, 0, 0.95, 1])

        plt.cla()

        pca = PCA(n_components=3)
        pca.fit(X_train)
        X_pca = pca.transform(X_train)

        y = np.choose(y_train, ['#1f77b4', '#ff7f0e'])
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

        plt.show()

    # For Timing Purposes
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    # Calculate Accuracy TODO: Try to use AUCROC for this!
    def calculate_accuracy(y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc
    
