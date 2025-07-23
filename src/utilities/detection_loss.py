

"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DOSEDSimpleLoss(nn.Module):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""
    
   
    def __init__(self,
                 number_of_classes,
                 device=torch.device("cuda"), 
                 ):
        super(DOSEDSimpleLoss, self).__init__()
        self.device = device
        self.number_of_classes = number_of_classes  # eventlessness

    def localization_loss(self, positive, localizations, localizations_target):
        # Localization Loss (Smooth L1)
        positive_expanded = positive.unsqueeze(positive.dim()).expand_as(localizations)
        loss_localization = F.smooth_l1_loss(localizations[positive_expanded].view(-1, 2),
                                             localizations_target[positive_expanded].view(-1, 2),
                                             size_average=False)
        return loss_localization

    def get_negative_index(self, positive, classifications, classifications_target):
        negative = (classifications_target == 0)
        return negative

    def get_classification_loss(self, index, classifications, classifications_target):
        index_expanded = index.unsqueeze(2).expand_as(classifications)

        loss_classification = F.cross_entropy(
            classifications[index_expanded.gt(0)
                            ].view(-1, self.number_of_classes),
            classifications_target[index.gt(0)],
            size_average=False
        )
        return loss_classification

    def forward(self, localizations, classifications, localizations_target, classifications_target):

        positive = classifications_target > 0
        negative = self.get_negative_index(positive, classifications, classifications_target)

        number_of_positive_all = positive.long().sum().float()
        number_of_negative_all = negative.long().sum().float()

        # loc loss
        loss_localization = self.localization_loss(positive, localizations, localizations_target)

        # + Classification loss
        loss_classification_positive = 0
        if number_of_positive_all > 0:
            loss_classification_positive = self.get_classification_loss(
                positive, classifications, classifications_target)

        # - Classification loss
        loss_classification_negative = 0
        if number_of_negative_all > 0:
            loss_classification_negative = self.get_classification_loss(
                negative, classifications, classifications_target)

        # Loss: sum
        loss_classification_positive_normalized = (loss_classification_positive /
                                                   number_of_positive_all)
        loss_classification_negative_normalized = (loss_classification_negative /
                                                   number_of_negative_all)
        loss_localization_normalized = loss_localization / number_of_positive_all

        return (loss_classification_positive_normalized,
                loss_classification_negative_normalized,
                loss_localization_normalized)



class DOSEDWorstNegativeMiningLoss(DOSEDSimpleLoss):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""

    def __init__(self,
                 number_of_classes,
                 device=torch.device("cuda"),
                 factor_negative_mining=3,
                 default_negative_mining=10,
                 ):
        super(DOSEDWorstNegativeMiningLoss, self).__init__(
            number_of_classes=number_of_classes,
            device=device)
        self.factor_negative_mining = factor_negative_mining
        self.default_negative_mining = default_negative_mining

    def get_negative_index(self, positive, classifications, classifications_target):
        batch = classifications.shape[0]
        number_of_default_events = classifications.shape[1]
        number_of_positive = positive.long().sum(1)
        number_of_negative = torch.clamp(number_of_positive * self.factor_negative_mining,
                                         min=self.default_negative_mining)
        number_of_negative = torch.min(number_of_negative,
                                       (number_of_default_events - number_of_positive))
        loss_softmax = -torch.log(nn.Softmax(1)(
            classifications.view(-1, self.number_of_classes)).gather(
            1, classifications_target.view(-1, 1))).view(batch, -1)
        loss_softmax[positive] = 0
        _, loss_softmax_descending_index = loss_softmax.sort(1, descending=True)
        _, loss_softmax_descending_rank = loss_softmax_descending_index.sort(1)
        negative = (loss_softmax_descending_rank <
                    number_of_negative.unsqueeze(1).expand_as(loss_softmax_descending_rank))
        return negative



class DOSEDRandomNegativeMiningLoss(DOSEDSimpleLoss):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""

    def __init__(self,
                 number_of_classes,
                 device=torch.device("cuda"), 
                 factor_negative_mining=3,
                 default_negative_mining=10,
                 ):
        super(DOSEDRandomNegativeMiningLoss, self).__init__(
            number_of_classes=number_of_classes,
            device=device)
        self.factor_negative_mining = factor_negative_mining
        self.default_negative_mining = default_negative_mining

    def get_negative_index(self, positive, classifications, classifications_target):
        number_of_default_events = classifications.shape[1]
        number_of_positive = positive.long().sum(1)
        number_of_negative = torch.clamp(number_of_positive * self.factor_negative_mining,
                                         min=self.default_negative_mining)
        number_of_negative = torch.min(number_of_negative,
                                       (number_of_default_events - number_of_positive))

        def pick_zero_random_index(tensor, size):
            result = torch.zeros_like(tensor)
            for index in np.random.choice(
                    (1 - tensor).nonzero().view(-1), size=size, replace=False):
                result[index] = 1
            return result
        random_negative_index = [pick_zero_random_index(line, int(number_of_negative[i]))
                                 for i, line in enumerate(torch.unbind(positive, dim=0))]
        negative = torch.stack(random_negative_index, dim=0)

        return negative


class DOSEDFocalLoss(DOSEDSimpleLoss):
    """Loss function inspired from https://github.com/amdegroot/ssd.pytorch"""

    def __init__(self,
                 number_of_classes,
                 device=torch.device("cuda"),
                 alpha=0.25,
                 gamma=2,
                 ):
        super(DOSEDFocalLoss, self).__init__(
            number_of_classes=number_of_classes,
            device=device)
        self.device = device
        self.number_of_classes = number_of_classes  # eventlessness
        self.alpha = alpha
        self.gamma = gamma

    def get_classification_loss(self, index, classifications, classifications_target):
        index_expanded = index.unsqueeze(2).expand_as(classifications)

        cross_entropy = F.cross_entropy(
            classifications[index_expanded.gt(0)
                            ].view(-1, self.number_of_classes),
            classifications_target[index.gt(0)],
            size_average=False,
            reduce=False
        )
        pt = torch.exp(-cross_entropy)
        loss_classification = (self.alpha * ((1 - pt) ** self.gamma) * cross_entropy).sum()
        return loss_classification
