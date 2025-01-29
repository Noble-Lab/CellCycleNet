'''
CellCycleNet uses images of individual, DAPI-stained nuclei to predict cell cycle phase (either G1 or S/G2).
There are two use cases for CellCycleNet: 1) using our pretrained model to predict on unlabeled data and 2) fine-tuning our pretrained model to fit the user's data (labels required).
In both use cases, users are required to input an image directory containing .tif files of single, segmented nuclei.

In use case #1, the workflow is:
    1. Instantiate the CellCycleNet object.
    2. call the .create_dataset() method with split_data=False to generate a single dataframe.
    3. call the .predict() method with with_labels=False.

In use case #2, the workflow is:
    1. Instantiate the CellCycleNet object.
    2. call the .create_dataset() method with split_data=True to generate train, validation, and test dataframes.
    3. call the .train() method
'''

####################################################################################################
####################################################################################################

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import torch
import re
import ast
from sklearn.model_selection import train_test_split
from tifffile import imread
from glob import glob
from cellcyclenet.unet3d.model import UNet3D
from cellcyclenet import models
from cellcyclenet.unet2d import UNet2D
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms 
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

####################################################################################################
####################################################################################################

class CCN_Dataset(Dataset):
    '''Create a class to hold the PyTorch Dataset, input to PyTorch Dataloader.'''
    def __init__(self, X, y, transform, lazy_load):
        self.X = X
        self.y = y
        self.transform = transform
        self.lazy_load = lazy_load

    def __len__(self):
        '''Denotes the total number of samples.'''
        return len(self.X)

    # define transform to convert image to tensor #
    tensor_transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    
    def __getitem__(self, index):
        '''Generates one image-label pair. Called when iterating over PyTorch dataloader.'''
        # if initialized with images already loaded (without lazy loading), just index to get image #
        if not self.lazy_load:
            image = self.X[index]

        # if initialized with image fns (lazy loading), load, normalize, and scale image #
        else:
            image = imread(self.X[index])

        # convert image to float 32 #
        X = np.asarray(image, dtype=np.float32)
        X = np.expand_dims(X, 0)

        # convert label to float 32 #
        label = self.y[index]
        y = np.asarray(label, dtype=np.float32)
        
        # apply transform for data augmentation, if specified when CCN_Dataset is called #
        if self.transform != None:
            X = self.transform(X) # apply image transformations

        # convert to PyTorch tensor #
        X = np.squeeze(X)
        X = self.tensor_transform(X)

        return X, y
    
####################################################################################################
####################################################################################################

class CellCycleNet:

    def __init__(self, state_dict_path=None, is_3d=True):
        # initialize device as GPU if available, otherwise CPU #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_3d = is_3d

        # initialize model architecture #
        if self.is_3d:
            self.model = UNet3D(in_channels=1, out_channels=1, is_segmentation=False, f_maps=32)
        else:
            self.model = UNet2D(in_channels=1, init_features=32)
        self.model = torch.nn.DataParallel(self.model)

        # if user does not provide a path to weights, load pretrained weights #
        if state_dict_path is None:
            if self.is_3d:
                state_dict_path = pkg_resources.files(models).joinpath('pretrained-model_3D.pt')
            else:
                state_dict_path = pkg_resources.files(models).joinpath('pretrained-model_2D.pt')

        # load model weights #
        if torch.cuda.is_available(): # load to GPU if available #
            state_dict = torch.load(state_dict_path)
        else: # otherwise, load to CPU #
            state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    ################################################################################################

    def create_dataset(self, dataframe, split_data=False, seed=15):
        '''
        Creates training, validation, and testing dataframes containing GT labels and image filenames for each nucleus.
        (NOTE: it is assumed that the order of labels in the dataframe is the same as order of images in image_dir.)
            Arguments:
                - image_dir [str] : path to directory of single nucleus images
                - dataframe [pd.DataFrame] : dataframe containing GT labels; if None, it is assumed that user wants to proceed without using labels (use case 1)
                - split_data [bool] : flag to determine if input data is split into train/val/test sets (use case 2) or just a single dataset (use case 1)
                - seed [int] : random seed to use for train/val/test split
            Outputs:
                - train, val, test [pd.DataFrame] : dataframe containing GT label (if inputted) + image filename for each nucleus
        '''
        ### FIXME small DF for debugging ###
        # dataframe = dataframe.iloc[::15]

        # if user wants to train their own model (split_data=True), split into three sets #
        if split_data:
            # Train / Val / Test split #
            train, val_test = train_test_split(dataframe, test_size=0.3, random_state=seed)
            val, test = train_test_split(val_test, test_size=0.33, random_state=seed)
            print(f'Created training (n = {len(train)}), validation (n = {len(val)}), and testing splits (n = {len(test)})...')
            return train, val, test
    
        # if user only wants to get predictions (split_data=False), return entire dataframe #
        else:
            print(f'Created dataset with {len(dataframe)} images...')
            return dataframe

    ################################################################################################

    def run_epoch(self, dataloader, is_train):
        '''
        Passes dataloader through the CCN a single time and computes performance metrics for training.
            Arguments:
                - dataloader [torch.utils.data.DataLoader] : dataloader to store image / label pairs
                - is_train [bool] : flag to determine if model weights are updated during run_epoch() call
            Outputs:
                - batch_acc [list] : running total accuracy for each batch
                - batch_loss [list] : loss for each batch
        '''
        # track training accuracy and loss per-batch over the epoch #
        batch_acc = []
        batch_loss = []
        running_count_correct = 0
        running_count_total = 0

        for k, (images, labels) in enumerate(dataloader):

            # get image / label pairs from dataloader and send to GPU
            images.to(self.device)
            labels.to(self.device)

            # for 3D images, reshape images to [batch, channel, Z, Y, X] #
            if self.is_3d:
                images = torch.swapaxes(images, 1, 2)
                images = torch.unsqueeze(images, 1)

            ### FORWARD PASS ###
            outputs = self.model(images)
            outputs = outputs.to('cpu')

            ### BACKWARD PASS ###
            loss = self.loss_fxn(outputs, torch.unsqueeze(labels, 1))
            if is_train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # send outputs to CPU, get predictions #
            preds = np.round(torch.sigmoid(outputs).detach().numpy())
            labels = np.asarray(torch.unsqueeze(labels, 1).to('cpu').detach().numpy())

            # track performance metrics #
            running_count_correct += np.equal(preds, labels).sum().item()
            running_count_total += len(labels)
            batch_acc.append(running_count_correct / running_count_total)
            batch_loss.append(loss.item())

            # FIXME debugging #
            # print(f'B{k} A{batch_acc[-1]:.3f}')
            # print(f'B{k} L{batch_loss[-1]:.3f}')
            # print('\n')

            # FIXME debugging #
            # if k > 10:
            #     break

        return np.asarray(batch_acc), np.asarray(batch_loss)

    ################################################################################################

    def train(self, train_df, val_df, n_epochs, batch_size=4, initial_LR=1e-5, transform=None, lazy_load=False, verbose=False):
        '''
        Trains CCN model.
            Arguments:
                - train_df [pd.DataFrame] : train set dataframe output by create_pytorch_datasets()
                - val_df [pd.DataFrame] : validation set dataframe output by create_pytorch_datasets()
                - n_epochs [int] : number of epochs to train model for
        '''
        # set best and last weights as CCN attribute #
        self.weights_best = None
        self.weights_last = None

        # initialize various things and stuff #
        torch.backends.cudnn.benchmark = True
        start_time = time()
        date_tag = datetime.now().strftime('%Y%m%d_%H:%M:%S')
        self.loss_fxn = torch.nn.BCEWithLogitsLoss() 
        batch = 0
        running_val_acc = []

        # initialize optimizer and scheduler for adjusting learning rate and weight decay during training #
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-1, lr=initial_LR)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.optimizer.zero_grad()

        # create dataloader for training data #
        print(f'Loading training (n = {len(train_df)}) images...')
        train_X_fn = train_df['filename'].values
        train_y = np.where(train_df['label'].values == 'G1', 0, 1)
        if lazy_load:
            train_dataloader = DataLoader(CCN_Dataset(train_X_fn, train_y, transform=transform, lazy_load=lazy_load),
                                                      batch_size=batch_size, shuffle=False)
        else:
            train_X = np.asarray([imread(fn) for fn in train_X_fn])
            train_dataloader = DataLoader(CCN_Dataset(train_X, train_y, transform=transform, lazy_load=lazy_load),
                                                      batch_size=batch_size, shuffle=False)
        
        # create dataloader for validation data #
        print(f'Loading validation (n = {len(val_df)}) images...')
        val_X_fn = val_df['filename'].values
        val_y = np.where(val_df['label'].values == 'G1', 0, 1)

        if lazy_load:
            val_dataloader = DataLoader(CCN_Dataset(val_X_fn, val_y, transform=None, lazy_load=lazy_load),
                                        batch_size=batch_size, shuffle=False)
        else:
            val_X = np.asarray([imread(fn) for fn in val_X_fn])
            val_dataloader = DataLoader(CCN_Dataset(val_X, val_y, transform=None, lazy_load=lazy_load),
                                        batch_size=batch_size, shuffle=False)

        # track val acc for each epoch to check for improvement #
        running_val_acc = []

        # iterate over epochs #
        for epoch in range(1, n_epochs+1):

            # run training set through model #
            train_batch_acc, train_batch_loss = self.run_epoch(train_dataloader, is_train=True)
            train_epoch_acc = np.average(train_batch_acc)
            train_epoch_loss = np.sum(train_batch_loss) / len(train_dataloader)
            if verbose:
                print(f'Train Acc.: (E{epoch}) = {train_epoch_acc:.3f}')
                print(f'Train Loss: (E{epoch}) = {train_epoch_loss:.3f} ')
                print('\n')

            # run validation set through model #
            with torch.no_grad():
                val_batch_acc, val_batch_loss = self.run_epoch(val_dataloader, is_train=False)
            val_epoch_acc = np.average(val_batch_acc)
            val_epoch_loss = np.sum(val_batch_loss) / len(val_dataloader)
            if verbose:
                print(f'Val Acc.: (E{epoch}) = {val_epoch_acc:.3f}')
                print(f'Val Loss: (E{epoch}) = {val_epoch_loss:.3f} ')
                print('\n')

            # save recent weights; save best weights if improved #
            # torch.save(self.model.state_dict(), f'{model_save_path}{date_tag}_last.pt')
            self.weights_last = self.model.state_dict()
            max_val_acc = max(running_val_acc) if running_val_acc else 0
            if val_epoch_acc > max_val_acc:
                # torch.save(self.model.state_dict(), f'{model_save_path}{date_tag}_best.pt')
                self.weights_best = self.model.state_dict()
        
        print(f'Finished training in {time() - start_time:.3f} seconds.')

    
    ################################################################################################

    def predict(self, dataframe, with_labels, decision_threshold=0.5):
        '''
        Generates predictions for image-label pairs.
            Arguments:
                - dataframe [pd.DataFrame] : dataframe containing image-label pairs to be predicted on
                - with_labels [bool] : flag to determine whether labels are expected in the input dataframe or not
                - decision_threshold [float] : decision threshold to use when turning probabilities into predictions
            Outputs:
                - dataframe [pd.DataFrame] : input dataframe with predictions, probabilties, and labels appended
        '''
        # Create empty lists for preds and probs #
        all_preds = []
        all_probs = []
        all_labels = []

        X_fn = dataframe['filename'].values
        y = np.where(dataframe['label'].values == 'G1', 0, 1) if with_labels else np.zeros(len(X_fn))
        dataloader = DataLoader(CCN_Dataset(X_fn, y, transform=None, lazy_load=True), batch_size=4, shuffle=False)

        # Run through network #
        with torch.no_grad():
            for images, labels in dataloader:
                # Send images and labels to device #
                images = images.to(self.device)
                labels = labels.to(self.device)

                # for 3D images, reshape to [batch, channels, Z, Y, X] #
                if self.is_3d:
                    images = torch.swapaxes(images, 1, 2)
                    images = torch.unsqueeze(images, 1)

                # Run inference #
                outputs = self.model(images)
                outputs = outputs.to('cpu')
                probs = torch.nn.functional.sigmoid(outputs).to('cpu').detach().numpy()
                preds = np.round(torch.sigmoid(outputs)).to('cpu').detach().numpy()

                all_preds.extend([int(pred[0]) for pred in preds])
                all_probs.extend([float(prob[0]) for prob in probs])
                all_labels.extend([int(label) for label in labels])

        # Append preds and probs to dataframe #
        dataframe['pred'] = all_preds
        dataframe['prob'] = all_probs
        if with_labels: dataframe['label'] = all_labels
        return dataframe
    
    ################################################################################################

    def plot_ROC(self, labels, preds, probs, var_to_return=None):
        '''
        Plots ROC curve for input classification results and returns requested metrics.
            Arguments:
                - labels [np.ndarray] : ground truth labels
                - preds [np.ndarray] :  predicted class
                - probs [np.ndarray] : predicted probability
                - var_to_return [list] : list of metrics to return (one of: accuracy, confusion_matrix, FPR, TPR, AUC)
            Returns:
                - metrics [dict] : dictionary of requested metrics (keys are defined by var_to_return, values are the metrics)
        '''
        # calculate AUC and TPR/FPR for ROC curve #
        AUC = roc_auc_score(labels, probs)
        FPR, TPR, _ = roc_curve(labels, probs)

        # calculate accuracy #
        cm = confusion_matrix(labels, preds)
        acc = np.round((cm[0,0]+cm[1,1]) / np.sum(cm), 3)

        # print metrics #
        print(f'Accuracy = {acc:.3f}')
        print(f'Confusion Matrix: \nTN: {cm[0,0]}\t  FP: {cm[0,1]}\nFN: {cm[1,0]}\t  TP: {cm[1,1]}')

        # plot
        plt.figure(figsize=(6,6))
        plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)')
        plt.plot(FPR, TPR, color='b', label=f'AUC = {AUC:.2f}')

        # plot decorations #
        plt.xlim = [-0.05, 1.05]
        plt.ylim = [-0.05, 1.05]
        plt.title(f'CellCycleNet', fontsize=20)
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.legend(loc='lower right')
        plt.show()

        # define dictionary of variables to return #
        var_dict = {'accuracy': acc,
                    'confusion_matrix': cm,
                    'FPR': FPR,
                    'TPR': TPR,
                    'AUC': AUC}

        # collate requested metrics to be returned #
        if var_to_return == None:
            pass
        else:
            metrics = {var: var_dict[var] for var in var_to_return}
            return metrics
    
    ################################################################################################

    def show_image(self, dataframe, index=None, with_preds=True, hide_plot=False, figsize=(4,4)):
        '''
        Plot an image of a single nucleus and its true and predicted labels.
            Arguments:
                - dataframe [pd.DataFrame] : dataframe to load image + label + prediction from
                - index [int] : index of dataframe to load image from (defaults to random index)
                - with_preds [bool] : if True, display predicted label; otherwise, show only GT label
                - hide_plot [bool] : if True, supress plotting of image; otherwise, show image
                - fig_size [tuple] : (x,y) dimensions of plotted image
        '''
        # if index not specified, set to random row of dataframe #
        if index == None: index = np.random.randint(0, len(dataframe))

        # load image from dataframe, get label and prediction #
        indexed_dataframe = dataframe.iloc[index, :]
        image = imread(indexed_dataframe['filename'])
        label = indexed_dataframe['label']
        if with_preds:
            pred = indexed_dataframe['pred']
            prob = indexed_dataframe['prob']

        # plot image #
        if not hide_plot:
            plt.figure(figsize=figsize)
            plt.axis('off')
            if self.is_3d:
                plt.imshow(np.max(image, axis=0))
            else:
                plt.imshow(image)
            plt.title(f'Index: {index} / Label: {label} / Pred: {pred} / Prob: {prob:.3f}', fontsize=10)
            plt.show()

    ################################################################################################

    def save_model(self, save_path=None):
        if save_path == None: save_path = f'{datetime.now().strftime("%Y%m%d_%H:%M:%S")}.pt'
        torch.save(self.weights_best, save_path)
