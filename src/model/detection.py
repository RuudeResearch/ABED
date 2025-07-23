"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""

import torch.nn as nn
import torch as th
import numpy as np
from utilities.non_maximum_suppression import non_maximum_suppression
from utilities.decode import decode

from sklearn.base import BaseEstimator


class Detection(nn.Module, BaseEstimator):
    """"""

    def __init__(self,
                 number_of_classes,
                 overlap_non_maximum_suppression,
                 top_k_non_maximum_suppression,
                 classification_threshold,
                 use_argmax=False
                 ):
        super(Detection, self).__init__()
        #super().__init__()
        self.number_of_classes = number_of_classes
        self.overlap_non_maximum_suppression = overlap_non_maximum_suppression
        self.top_k_non_maximum_suppression = top_k_non_maximum_suppression
        self.classification_threshold = classification_threshold
        self.use_argmax = use_argmax

    def forward(self, localizations, classifications, detection_threshold,localizations_default):
        batch = localizations.size(0)

        if len(detection_threshold)!=4:
            z1=1
        else:
            z1=2

        scores_ground = nn.Softmax(dim=2)(classifications)
        cuda0 = th.device('cuda:0')

        indices1 = th.tensor([0 ,1]).cuda()
        indices2 = th.tensor([0 ,2]).cuda()
        indices3 = th.tensor([0 ,3]).cuda()
        indices4 = th.tensor([0 ,4]).cuda()

        a1 = th.index_select(classifications, 2, indices1)
        a2 = th.index_select(classifications, 2, indices2)
        a3 = th.index_select(classifications, 2, indices3)
        a4 = th.index_select(classifications, 2, indices4)

        scores1 = nn.Softmax(dim=2)(a1)
        scores2 = nn.Softmax(dim=2)(a2)
        scores3 = nn.Softmax(dim=2)(a3)
        scores4 = nn.Softmax(dim=2)(a4)

        idx0 = th.tensor([0]).cuda()
        idx1 = th.tensor([1]).cuda()

        col0=th.index_select(scores_ground,2,idx0)
        col1 = th.index_select(scores1, 2, idx1)
        col2 = th.index_select(scores2, 2, idx1)
        col3 = th.index_select(scores3, 2, idx1)
        col4 = th.index_select(scores4, 2, idx1)

        if z1==1:
            
            #scores = th.cat((col0,col1, col2,col3,col4), 2)
            scores = scores_ground

        elif z1==2:
            

            scores122 = th.cat((col1/detection_threshold[0], col2/detection_threshold[1],col3/detection_threshold[2],col4/detection_threshold[3]), 2)
            scores122 = scores122.cpu()

            scores122_actual = th.cat((col1, col2 , col3 ,col4), 2)
            scores122_actual = scores122_actual.cpu()
            scores122np_actual = scores122_actual.detach().numpy()
            scores122np_actual = scores122np_actual[0, :, :]


            
            scores122np = scores122.detach().numpy()
            scores122np = scores122np[0, :, :]
            scores122np_int = (scores122np == scores122np.max(axis=1)[:, None]).astype(int)

            scores122np_actual = np.float32(scores122np_actual*scores122np_int)
            scores122_th = th.tensor(scores122np_actual)
            scores122_th = scores122_th.unsqueeze(0)
            
            col0 = col0.cpu()
            col0 = col0.detach().numpy()
            col0 = col0[0, :, :]

            col0 = th.tensor(col0)
            col0 = col0.unsqueeze(0)

            #scores = th.cat((col0,scores122_th),2)
            
            scores = scores_ground




        results = []

        if self.use_argmax:
            _, idx_label = scores.max(dim=-1)
            for i in range(batch):
                result = []
                localization_decoded = decode(
                    localizations[i].data, localizations_default)

                for class_index in range(1, self.number_of_classes):

                    # check that some events are annotated
                    # ie check that there are labels different from 0
                    if (idx_label[i] == class_index).data.float().sum() == 0:
                        continue

                    (idx_label[i] == class_index)

                    # change appears here
                    mask = (idx_label[i] == class_index).data.long().nonzero()

                    scores_batch_class = scores[i, :, class_index].data
                    scores_batch_class_selected = scores_batch_class[
                        mask.squeeze()]

                    localizations_decoded_selected = localization_decoded[
                        mask, :].view(-1, 2)
                    
                    #print(localizations_decoded_selected)
                    #print(scores_batch_class_selected)
                    
                    result.extend(
                        [[x[0].cpu(), x[1].cpu(), class_index - 1, x[2].cpu()]
                         for x in non_maximum_suppression(
                            localizations_decoded_selected,
                            scores_batch_class_selected,
                            #scores[i, :, :].data,
                            overlap=self.overlap_non_maximum_suppression,
                            top_k=self.top_k_non_maximum_suppression)])
                results.append(result)

        else:
            for i in range(batch):
                result = []
                localization_decoded = decode(
                    localizations[i].data, localizations_default)
                for class_index in range(1, self.number_of_classes):
                    scores_batch_class = scores[i, :, class_index].data
                    scores_batch_class_selected = scores_batch_class[
                        scores_batch_class > self.classification_threshold]
                    scores_selected = scores[:,scores_batch_class > self.classification_threshold,:]
                    
                    if len(scores_batch_class_selected) == 0:
                        continue
                    
                    localizations_decoded_selected = localization_decoded[
                        (scores_batch_class > self.classification_threshold)
                        .unsqueeze(1).expand_as(localization_decoded)
                    ].view(-1, 2)
                    
                    #print(localizations_decoded_selected)
                    #print(scores_batch_class_selected)
                    #print(scores)
                    #print(self.overlap_non_maximum_suppression)
                    #print(self.top_k_non_maximum_suppression)
                    
                    result.extend(
                        [[x[0].cpu(), x[1].cpu(), class_index - 1, x[2].cpu(), scores_selected[:, x[3], :].detach().cpu()]
                         for x in non_maximum_suppression(
                         localizations_decoded_selected,
                         scores_batch_class_selected,
                         #scores[i, :, :].data,
                         overlap=self.overlap_non_maximum_suppression,
                         top_k=self.top_k_non_maximum_suppression)])
                results.append(result)

        return results
