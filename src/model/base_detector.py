# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:21:57 2021

@author: Magnus Ruud Kjær
"""
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

# event detection imports
from utilities.datasets_utils import collate
from model.detection_loss import *

from utilities.matching_utils import *


class BaseDetector(nn.Module):

    def __init__(self,
                 n_classes=3,
                 num_workers=10, shuffle=True, pin_memory=False,
                 batch_size=18, epochs=100,
                 histories_path=None, weights_path=None,
                 threshold_overlap=0.5, factor_negative_mining=3,
                 default_negative_mining=10, negative_mining_mode="worst",
                 lr=1e-4, momentum=0.9, patience=30,
                 lr_decrease_patience=5, lr_decrease_factor=2.,
                 loss="simple",
                 loss_alpha=0.25, loss_gamma=2,
                 weight_loc_loss=1,
                 partial_eval=-1):

        super(BaseDetector, self).__init__()
        self.sizes = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.epochs = epochs
        self.histories_path = histories_path
        self.weights_path = weights_path

        self.threshold_overlap = threshold_overlap
        self.factor_negative_mining = factor_negative_mining
        self.default_negative_mining = default_negative_mining
        self.negative_mining_mode = negative_mining_mode
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma

        self.lr = lr
        self.momentum = momentum
        self.patience = patience
        self.lr_decrease_patience = lr_decrease_patience
        self.lr_decrease_factor = lr_decrease_factor

        # eventness, real events
        self.n_classes = n_classes + 1
        self.weight_loc_loss = weight_loc_loss

        # evaluation parameters
        self.partial_eval = partial_eval

        self.localizations_default = []

        self.history = None
        self.model_ = None

        # matching parameters
        self.matching_parameters = {
            "method": "new",
            "overlap": 0.4
        }

        # loss parameter
        self.loss = loss
        if self.loss == "simple":
            self.loss_parameters = {
                "number_of_classes": self.n_classes,
            }

            self.criterion = DOSEDSimpleLoss

        elif self.loss == "worst_negative_mining":
            self.loss_parameters = {
                "number_of_classes": self.n_classes,
                "factor_negative_mining": self.factor_negative_mining,
                "default_negative_mining": self.default_negative_mining,
            }

            self.criterion = DOSEDWorstNegativeMiningLoss

        elif self.loss == "random_negative_mining":
            self.loss_parameters = {
                "number_of_classes": self.n_classes,
                "factor_negative_mining": self.factor_negative_mining,
                "default_negative_mining": self.default_negative_mining,
            }

            self.criterion = DOSEDRandomNegativeMiningLoss

        elif self.loss == "focal":
            self.loss_parameters = {
                "number_of_classes": self.n_classes,
                "alpha": self.loss_alpha,
                "gamma": self.loss_gamma,
            }

            self.criterion = DOSEDFocalLoss

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def predict(self, x):
        localizations, classifications = self.forward(x)

        return localizations, classifications

    def predict_generator(
            self, test_gen,
            detection_threshold=[0.9, 0.8, 0.7, 0.6, 0.5]):

        sfreq = test_gen.df.fs.values[0]
        window = test_gen.window

        # binary vector for true labels
        true = np.zeros(
            (self.n_classes - 1,
             test_gen.df.n_times.values[0]))

        # binary vector of pred labels
        # shape n_thres, n_classes, n_times
        pred = np.zeros(
            (len(detection_threshold), self.n_classes - 1,
             test_gen.df.n_times.values[0]))
        
        # prob = np.zeros(
        #     (self.n_classes - 1,
        #      test_gen.df.n_times.values[0]))
        
        prob = np.zeros(
            (8, len(detection_threshold), self.n_classes,
             test_gen.df.n_times.values[0]))
        
        events_df = pd.DataFrame([],columns = ['start_sample', 'stop_sample', 'event', 'p_none','p_OA','p_CA','p_HYPO','p_NAD'])
        
        for idx_sample, (x, y) in enumerate(tqdm(test_gen)):

            # true vector
            y = y.numpy()
            if y.shape[0] != 0:
                for idx_event in range(y.shape[0]):
                    start = (y[idx_event, 0] + idx_sample/2) * window
                    end = (y[idx_event, 1] + idx_sample/2) * window
                    idx_class = int(y[idx_event, 2])

                    idx_start = int(start * sfreq)
                    idx_end = int(end * sfreq)

                    true[idx_class, idx_start:idx_end] = 1

            # pred vector
            x_ = x.unsqueeze(0)
            #z_loc, z_clf = self.predict(Variable(x_).cuda())
            z_loc, z_clf = self.predict(Variable(x_).to(self.device)) #update test

            # detection here
            for idx_thres, thres in enumerate(detection_threshold):
                # print(idx_thres)
                self.detector.set_params(classification_threshold=thres)

                # z = self.detector(
                #     z_loc, z_clf, detection_threshold,self.localizations_default.cuda())
                z = self.detector(
                    z_loc, z_clf, detection_threshold,self.localizations_default.to(self.device)) #update test
                # print(z)
                                      
                # print(z.shape)
                # print(type(z))
                z = np.asarray(z, dtype=object)
                if z.shape[1] != 0:
                    for idx_event in range(z.shape[1]):
                        if z[0, idx_event, 0] < 30/window:# and z[0, idx_event, 1] < 60/window:
                            continue
                            # print('too early')
                        elif z[0, idx_event, 1] > 210/window:# and z[0, idx_event, 0] > 180/window:
                            continue
                            # print('too late')
                        start = (z[0, idx_event, 0] + idx_sample/2) * window
                        end = (z[0, idx_event, 1] + idx_sample/2) * window
                        idx_class = int(z[0, idx_event, 2])
                        # print(f'idx_thres_{idx_thres}_thres_{thres}_idx_class_{idx_class}_z.shape_{z.shape}')

                        idx_start = int(start * sfreq)
                        idx_end = int(end * sfreq)

                        pred[idx_thres, idx_class, idx_start:idx_end] = 1

                        # if idx_class == idx_thres:
                        #     prob[0, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                        
                        # if overlap then get the middle part of window get z 

                        event_data = {
                            'start_sample': [idx_start],
                            'stop_sample': [idx_end],
                            'event': [idx_class],
                            'p_none': [z[0, idx_event, 4][0][0]],
                            'p_OA': [z[0, idx_event, 4][0][1]],
                            'p_CA': [z[0, idx_event, 4][0][2]],
                            'p_HYPO': [z[0, idx_event, 4][0][3]],
                            'p_NAD': [z[0, idx_event, 4][0][4]]
                        }

                        # Create a DataFrame from the new data
                        event_data_df = pd.DataFrame(event_data)
                        events_df = pd.concat([events_df, event_data_df], ignore_index=True)

                        if prob[0, idx_thres, :, idx_start:idx_end].sum() == 0:
                            prob[0, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                        elif prob[1, idx_thres, :, idx_start:idx_end].sum() == 0:
                            prob[1, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                        elif prob[2, idx_thres, :, idx_start:idx_end].sum() == 0:
                            prob[2, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                        elif prob[3, idx_thres, :, idx_start:idx_end].sum() == 0:
                            prob[3, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                        elif prob[4, idx_thres, :, idx_start:idx_end].sum() == 0:
                            prob[4, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                        elif prob[5, idx_thres, :, idx_start:idx_end].sum() == 0:
                            prob[5, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                        elif prob[6, idx_thres, :, idx_start:idx_end].sum() == 0:
                            prob[6, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                        elif prob[7, idx_thres, :, idx_start:idx_end].sum() == 0:
                            prob[7, idx_thres, :, idx_start:idx_end] = z[0, idx_event, 4].T
                            
                        #prob[:, idx_start:idx_end] = np.repeat(np.expand_dims(np.array([z[0, idx_event, 3].cpu().item(),z[0, idx_event, 4].cpu().item(),z[0, idx_event, 5].cpu().item()]),1), idx_end-idx_start,axis=1)
                        
                        #prob = 0
                        
        return true, pred, prob, events_df

    def fit_generator(self, train_gen, val_gen):

        #if params is None: #Don't know if these 3 lines should be here
        #params = self.parameters()

        #self.cuda()
        self.to(self.device) #update test

        dataloader_parameters_train = {
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "collate_fn": collate,
            "pin_memory": self.pin_memory,
            "batch_size": self.batch_size
        }
        dataloader_parameters_val = {
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "collate_fn": collate,
            "pin_memory": self.pin_memory,
            "batch_size": self.batch_size
        }

        train_loader = DataLoader(train_gen, **dataloader_parameters_train)
        val_loader = DataLoader(val_gen, **dataloader_parameters_val)

        # criterion
        #device = torch.device("cuda")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        criterion = self.criterion(**self.loss_parameters)

        print("\n\nusing {}".format(self.loss))

        if self.matching_parameters["method"] == "new":
            matching = match_events_localization_to_default_localizations
        else:
            matching = match_events_localization_to_default_localizations_old

        current_lr = self.lr
        optimizer = optim.Adam(     #should be SGD
            self.parameters(),weight_decay = 0.00000001, #before was low weight deacy
            lr=current_lr)#, momentum=self.momentum)#before was without momentum

        if self.partial_eval != -1:
            print("\nPerforming eval every {} batches".format(
                self.partial_eval))

        history = dict(train_clf=[], train_loc=[], val_clf=[], val_loc=[])
        best_net = copy.deepcopy(self)
        best_loss_val = np.infty
        waiting = 0
        waiting_lr_decrease = 0
        
        for epoch in range(self.epochs):

            # training loop
            print("\nStarting epoch : {} / {}".format(epoch + 1, self.epochs))

            bar_train = tqdm(train_loader)
            self.train()
            # scheduler.step()

            train_epoch_clf = []
            train_epoch_loc = []
            print(bar_train)
            for idx_batch, batch in enumerate(bar_train):

                # lr decrease included in training loop
                if waiting_lr_decrease == self.lr_decrease_patience:
                    print("\nReducing lr {} -> {}".format(
                        current_lr, current_lr / self.lr_decrease_factor))

                    current_lr /= self.lr_decrease_factor
                    optimizer = optim.Adam(
                        self.parameters(),weight_decay = 0.00000001, #before was low weight deacy
                        lr=current_lr)#, momentum=self.momentum)#before was without momentum
                    waiting_lr_decrease = 0
                    self.train()

                eeg, events = batch
                #x = eeg.cuda() 
                x = eeg.to(device)
                #print(x)
                #print(x.shape)

                optimizer.zero_grad()
                

                # step 1: forward pass #update test  
                locs, clfs = self.forward(x)

                # step 2: matching
                localizations_target, classifications_target = matching(
                    localizations_default=self.localizations_default,
                    events=events,
                    threshold_overlap=self.matching_parameters["overlap"])
                localizations_target = localizations_target.to(device)
                classifications_target = classifications_target.to(device)

                # step 3: loss
                train_clf_pos_loss, train_clf_neg_loss, train_loc_loss = (
                    criterion(locs,
                              clfs,
                              localizations_target,
                              classifications_target))

                loss = (train_clf_pos_loss + train_clf_neg_loss) + self.weight_loc_loss * train_loc_loss
                loss.backward()
                optimizer.step()

                # step 4: monitoring
                train_epoch_clf.append(
                    train_clf_pos_loss.item() + train_clf_neg_loss.item())
                train_epoch_loc.append(train_loc_loss.item())

                train_epoch_clf_ = np.asarray(train_epoch_clf)
                m_clf = np.mean(
                    train_epoch_clf_[np.isfinite(train_epoch_clf_)])
                train_epoch_loc_ = np.asarray(train_epoch_loc)
                m_loc = np.mean(
                    train_epoch_loc_[np.isfinite(train_epoch_loc_)])

                bar_train.set_description(
                    'clf: {:.4f} | loc: {:.4f}'.format(
                        m_clf,
                        m_loc))

                # partial eval (useful when working with long epochs)
                if self.partial_eval != -1:
                    if idx_batch != 0:
                        if idx_batch % self.partial_eval == 0:

                            history["train_loc"].append(m_loc)
                            history["train_clf"].append(m_clf)

                            bar_val = tqdm(val_loader)
                            self.eval()

                            val_epoch_clf = []
                            val_epoch_loc = []

                            for idx_batch_, batch in enumerate(bar_val):

                                eeg, events = batch
                                #x = eeg.cuda() 
                                x = eeg.to(device)
                                #print(x.shape)

                                # step 1: forward pass
                                locs, clfs = self.forward(x)

                                # step 2: matching
                                localizations_target, classifications_target = matching(
                                    localizations_default=self.localizations_default,
                                    events=events,
                                    threshold_overlap=self.matching_parameters["overlap"])
                                localizations_target = localizations_target.to(device)
                                classifications_target = classifications_target.to(device)

                                # step 3: loss
                                val_clf_pos_loss, val_clf_neg_loss, val_loc_loss = (
                                    criterion(locs,
                                              clfs,
                                              localizations_target,
                                              classifications_target))

                                loss = (val_clf_pos_loss + val_clf_neg_loss) + self.weight_loc_loss * val_loc_loss

                                # step 4: monitoring
                                val_epoch_clf.append(
                                    val_clf_neg_loss.item() + val_clf_pos_loss.item())
                                val_epoch_loc.append(val_loc_loss.item())

                                val_epoch_clf_ = np.asarray(val_epoch_clf)
                                m_clf = np.mean(
                                    val_epoch_clf_[np.isfinite(val_epoch_clf_)])
                                val_epoch_loc_ = np.asarray(val_epoch_loc)
                                m_loc = np.mean(
                                    val_epoch_loc_[np.isfinite(val_epoch_loc_)])

                                bar_val.set_description(
                                    'clf: {:.4f} | loc: {:.4f}'.format(
                                        m_clf, m_loc))

                           
                            history["val_loc"].append(m_loc)
                            history["val_clf"].append(m_clf)
                            

                            # early stopping
                            val_loss_epoch = (self.weight_loc_loss * m_loc) + m_clf
                            if val_loss_epoch < best_loss_val:
                                print("\n\nval loss improved: {:.4f} -> {:.4f}\n".format(
                                    best_loss_val, val_loss_epoch))
                                best_loss_val = val_loss_epoch
                                best_net = copy.deepcopy(self)
                                waiting = 0
                                waiting_lr_decrease = 0
                            else:
                                print("\n\nval loss did not improved: {:.4f} < {:.4f}\n".format(
                                    best_loss_val, val_loss_epoch))
                                waiting += 1
                                waiting_lr_decrease += 1

                            if waiting == self.patience:
                                break

                            self.train()
                            train_epoch_clf = []
                            train_epoch_loc = []



            history["train_loc"].append(m_loc)
            history["train_clf"].append(m_clf)

            # validation loop
            bar_val = tqdm(val_loader)
            self.eval()

            val_epoch_clf = []
            val_epoch_loc = []

            for idx_batch, batch in enumerate(bar_val):

                eeg, events = batch
                #x = eeg.cuda() 
                x = eeg.to(device)

                # step 1: forward pass
                locs, clfs = self.forward(x)

                # step 2: matching
                localizations_target, classifications_target = matching(
                    localizations_default=self.localizations_default,
                    events=events,
                    threshold_overlap=self.matching_parameters["overlap"])
                localizations_target = localizations_target.to(device)
                classifications_target = classifications_target.to(device)

                # step 3: loss
                val_clf_pos_loss, val_clf_neg_loss, val_loc_loss = (
                    criterion(locs,
                              clfs,
                              localizations_target,
                              classifications_target))

                loss = (val_clf_pos_loss + val_clf_neg_loss) + self.weight_loc_loss * val_loc_loss

                # step 4: monitoring
                val_epoch_clf.append(
                    val_clf_neg_loss.item() + val_clf_pos_loss.item())
                val_epoch_loc.append(val_loc_loss.item())

                val_epoch_clf_ = np.asarray(val_epoch_clf)
                m_clf = np.mean(
                    val_epoch_clf_[np.isfinite(val_epoch_clf_)])
                val_epoch_loc_ = np.asarray(val_epoch_loc)
                m_loc = np.mean(
                    val_epoch_loc_[np.isfinite(val_epoch_loc_)])

                bar_val.set_description(
                    'clf: {:.4f} | loc: {:.4f}'.format(
                        m_clf, m_loc))

            history["val_loc"].append(m_loc)
            history["val_clf"].append(m_clf)

            # early stopping
            val_loss_epoch = (self.weight_loc_loss * m_loc) + m_clf
            if val_loss_epoch < best_loss_val:
                print("val loss improved: {:.4f} -> {:.4f}".format(
                    best_loss_val, val_loss_epoch))
                best_loss_val = val_loss_epoch
                best_net = copy.deepcopy(self)
                waiting = 0
                waiting_lr_decrease = 0
            else:
                print("val loss did not improve: {:.4f} < {:.4f}".format(
                    best_loss_val, val_loss_epoch))
                waiting += 1
                waiting_lr_decrease += 1

            if waiting == self.patience:
                break

        history = pd.DataFrame(history)
        history["epoch"] = np.arange(history.shape[0])

        if self.weights_path is not None:
            torch.save(best_net.state_dict(), self.weights_path)

        if self.histories_path is not None:
            history.to_csv(self.histories_path)

        self = best_net
        self.history = history

        return val_loss_epoch
