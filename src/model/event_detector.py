# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:06:57 2021

@author: Magnus Ruud Kjær
"""

import numpy as np
from collections import OrderedDict

# torch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torchvision.models.resnet import BasicBlock, ResNet

# paper imports
from model.detection import Detection
from model.base_detector import BaseDetector
from model.detection_loss import *

from model.event_detector_resnet import ResNet, BasicBlock, Bottleneck

def get_overlerapping_default_events(
        window_size, default_event_sizes, factor_overlap=2):
    window_size = window_size
    default_event_sizes = default_event_sizes
    factor_overlap = factor_overlap
    default_events = []

    # for default_event_size in default_event_sizes20:
    #    overlap = default_event_size / factor_overlap
    #    number_of_default_events20 = int(window_size / overlap)
    #    default_events20.extend(
    #        [(overlap * (0.5 + i) / window_size,
    #          default_event_size / window_size)
    #         for i in range(number_of_default_events20)]
    #    )

    # for default_event_size in default_event_sizes30:
    #    overlap = default_event_size / factor_overlap
    #    number_of_default_events30 = int(window_size / overlap)
    #    default_events30.extend(
    #        [(overlap * (0.5 + i) / window_size,
    #          default_event_size / window_size)
    #         for i in range(number_of_default_events30)]
    #    )

    for default_event_size in default_event_sizes:
        overlap = default_event_size / factor_overlap
        number_of_default_events = int(window_size / overlap)
        default_events.extend(
            [(overlap * (0.5 + i) / window_size,
              default_event_size / window_size)
             for i in range(number_of_default_events)]
        )

    # default_events.extend(default_events20)
    # default_events.extend(default_events30)

    default_events.sort()

    return torch.Tensor(default_events)


class EventDetector(BaseDetector):

    def __init__(self, n_times=7680, n_channels=4, fs=256,  # n_times er 30s n_channels changed from 1
                 n_classes=3, n_freq_channels=4, freq_factor=[1,1,1,1],
                 overlap_non_maximum_suppression=0.4,
                 top_k_non_maximum_suppression=200,
                 classification_threshold=0.7,
                 num_workers=10, shuffle=True, pin_memory=True,
                 batch_size=18, epochs=100,  # was 32   epochs was 100
                 histories_path=None, weights_path=None,
                 threshold_overlap=0.5, factor_negative_mining=3,
                 default_negative_mining=10, negative_mining_mode="worst",
                 lr=1e-4, momentum=0.9, patience=30,
                 lr_decrease_patience=5, lr_decrease_factor=10,
                 loss="simple",
                 loss_alpha=0.25, loss_gamma=2,
                 k_max=9, max_pooling=2,
                 default_event_sizes=[1 * 256], factor_overlap=4,
                 weight_loc_loss=1,
                 partial_eval=-1, dropout=0, linearlayer=0,
                 RES_architecture=[2,2,2,2]):
      
        super(EventDetector, self).__init__()
        
        self.linearlayer = linearlayer
        self.dropout = dropout
        self.n_freq_channels = n_freq_channels
        self.freq_factor = freq_factor

        self.n_times = n_times
        self.n_channels = n_channels
        self.fs = fs
        self.overlap_non_maximum_suppression = overlap_non_maximum_suppression
        self.top_k_non_maximum_suppression = top_k_non_maximum_suppression
        self.classification_threshold = classification_threshold

        self.threshold_overlap = threshold_overlap
        self.factor_negative_mining = factor_negative_mining
        self.default_negative_mining = default_negative_mining
        self.negative_mining_mode = negative_mining_mode
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.max_pooling = max_pooling

        self.lr = lr
        self.momentum = momentum
        self.patience = patience
        self.lr_decrease_patience = lr_decrease_patience
        self.lr_decrease_factor = lr_decrease_factor

        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.epochs = epochs
        self.histories_path = histories_path
        self.weights_path = weights_path
        self.k_max = k_max
        self.factor_overlap = factor_overlap
        self.weight_loc_loss = weight_loc_loss

        self.partial_eval = partial_eval

        # eventness, real events
        self.n_classes = n_classes + 1
        self.history = None

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

        self.detector = Detection(
            number_of_classes=self.n_classes,
            overlap_non_maximum_suppression=self.overlap_non_maximum_suppression,
            top_k_non_maximum_suppression=self.top_k_non_maximum_suppression,
            classification_threshold=self.classification_threshold)

        self.localizations_default = get_overlerapping_default_events(
            window_size=n_times,
            default_event_sizes=default_event_sizes,
            factor_overlap=self.factor_overlap
        )

        if self.n_channels == 6:
            self.spatial_filtering = nn.Conv2d(
                1, self.n_channels - 2, (self.n_channels - 2, 1))

        elif self.n_channels != 1:
            self.spatial_filtering = nn.Conv2d(
                1, self.n_channels, (self.n_channels, 1))

        actual_chn = self.n_channels - self.n_freq_channels
        if self.n_freq_channels > 0.5:
            actual_chn = actual_chn + 1

        

            # scale block
        # self.block_scale = nn.Sequential(
        #    OrderedDict([
        #        ("conv_{}".format(1), nn.Conv1d(
        #            in_channels=1,
        #            out_channels=1,
        #            kernel_size=(1, 1))),
        #        ("padding_{}".format(1),
        #         nn.ConstantPad2d([0, 0, 0, 0], 0)),
        #        ("batchnorm_{}".format(1),
        #         nn.BatchNorm2d(0 * (2 ** 1))),
        #        ("relu_{}".format(1), nn.ReLU()),
        #        ("max_pooling_{}".format(1),
        #         nn.MaxPool2d(kernel_size=(1, 1)))]))

        # self.block_scale = nn.Conv2d(
        #         self.n_channels - self.n_freq_channels,self.n_channels - self.n_freq_channels, (1,1))

        # self.block_scale2 = nn.Conv2d(
        #         self.n_channels - self.n_freq_channels+1,self.n_channels - self.n_freq_channels+1, (1,1))
        '''
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        #num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_channels: int=5,
        inwindow: int=120,
        sfreq: int=8
        '''
        #resnet block 
        self.resnet = ResNet(#block = BasicBlock,
                             block = Bottleneck,
                             #layers = [3, 4, 6, 3],
                             layers = RES_architecture,
                             n_channels = len(freq_factor),
                             inwindow = n_times/fs,
                             sfreq = fs,
                             drop = self.dropout)
        
        # first block
        self.block1_1 = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_{}".format(1), nn.Conv1d(
                            in_channels=1,
                            out_channels=8,
                            kernel_size=(1, 3))),
                        ("padding_{}".format(1),
                         nn.ConstantPad2d([1, 1, 0, 0], 0)),
                        ("batchnorm_{}".format(1),
                         nn.BatchNorm2d(4 * (2 ** 1))),
                        ("relu_{}".format(1), nn.ReLU()),
                        ("max_pooling_{}".format(1),
                         nn.MaxPool2d(kernel_size=(1, self.max_pooling)))
                    ])
                ) for k in range(  len(self.freq_factor))
            ]
        )

        self.block1_12 = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_{}".format(6), nn.Conv2d(
                            in_channels=4 * (2 ** (2 - 1)),
                            out_channels=4 * (2 ** 2),
                            kernel_size=(1, 3))),
                        ("padding_{}".format(6),
                         nn.ConstantPad2d([1, 1, 0, 0], 0)),
                        ("batchnorm_{}".format(6),
                         nn.BatchNorm2d(4 * (2 ** (2)))),
                        ("relu_{}".format(6), nn.ReLU()),
                        ("max_pooling_{}".format(6),
                         nn.MaxPool2d(kernel_size=(1, self.max_pooling)))])) for k in range(  len(self.freq_factor))
            ]
        )

        self.block1_13 = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_{}".format(11), nn.Conv2d(
                            in_channels=4 * (2 ** (3 - 1)),
                            out_channels=4 * (2 ** 3),
                            kernel_size=(1, 3))),
                        ("padding_{}".format(11),
                         nn.ConstantPad2d([1, 1, 0, 0], 0)),
                        ("batchnorm_{}".format(11),
                         nn.BatchNorm2d(4 * (2 ** (3)))),
                        ("relu_{}".format(11), nn.ReLU()),
                        ("max_pooling_{}".format(11),
                         nn.MaxPool2d(kernel_size=(1, self.max_pooling)))
                    ])
                ) for k in range(  len(self.freq_factor))
            ]
        )

        # other blocks
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_{}".format(20 + k), nn.Conv1d(
                            in_channels=4 * (2 ** (k - 1)),
                            out_channels=4 * (2 ** k),
                            kernel_size=(1, 3))),
                        ("padding_{}".format(20 + k),
                         nn.ConstantPad2d([1, 1, 0, 0], 0)),
                        ("batchnorm_{}".format(20 + k),
                         nn.BatchNorm2d(4 * (2 ** k))),
                        ("relu_{}".format(20 + k), nn.ReLU()),
                        ("max_pooling_{}".format(20 + k),
                         nn.MaxPool2d(kernel_size=(1, self.max_pooling)))
                    ])
                ) for k in [i for i in range(4, self.k_max+1) for _ in range(  len(self.freq_factor))]
            ]
        )


        extra = 0
        self.blocks_ekstra = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_{}".format(20 + k), nn.Conv1d(
                            in_channels=4 * (2 ** (self.k_max+extra)),
                            out_channels=4 * (2 ** self.k_max+extra),
                            kernel_size=(1, 3))),
                        ("padding_{}".format(20 + self.k_max+extra),
                         nn.ConstantPad2d([1, 1, 0, 0], 0)),
                        ("batchnorm_{}".format(20 + self.k_max+extra),
                         nn.BatchNorm2d(4 * (2 ** self.k_max+extra))),
                        ("relu_{}".format(20 + self.k_max+extra), nn.ReLU()),
                        ("max_pooling_{}".format(20 + self.k_max+extra),
                         nn.MaxPool2d(kernel_size=(1, self.max_pooling)))
                    ])
                ) for k in range(int(sum(np.log2(self.freq_factor))))
            ]
        )
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.rnn = nn.LSTM(2048,1024,2,bidirectional=True)



        # other blocks
        # self.blocks_2d = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             OrderedDict([
        #                 ("conv_{}".format(k), nn.Conv2d(
        #                     in_channels=4 * (2 ** (k - 1)),
        #                     out_channels=4 * (2 ** k),
        #                     kernel_size=(1, 3))),
        #                 ("padding_{}".format(k),
        #                  nn.ConstantPad2d([1, 1, 0, 0], 0)),
        #                 ("batchnorm_{}".format(k),
        #                  nn.BatchNorm2d(4 * (2 ** k))),
        #                 ("relu_{}".format(k), nn.ReLU()),
        #                 ("max_pooling_{}".format(k),
        #                  nn.MaxPool2d(kernel_size=(1, self.max_pooling)))
        #             ])
        #         ) for k in range(self.k_max-5, self.k_max)
        #     ]
        # )
        '''
        self.reduction = nn.Sequential(
            nn.Conv2d(
                #in_channels=512,
                in_channels=2048,
                out_channels=(self.n_classes + 2) * len(self.localizations_default),
                kernel_size=(1, int(n_times/(2**5))))
        )
        '''
        
        
        
        
        #self.fc_loc = nn.Linear((self.n_classes + 2) * len(self.localizations_default), 2 * len(self.localizations_default))
        
        #self.fc_clf = nn.Linear((self.n_classes + 2) * len(self.localizations_default), self.n_classes * len(self.localizations_default))


        self.localization = nn.Sequential(
            nn.Conv2d(
                #in_channels=512,
                in_channels=2048,
                #in_channels=128,
                out_channels=2 * len(self.localizations_default),
                kernel_size=(1, int(n_times/(2**5))))
        )

        self.classification = nn.Sequential(
            nn.Conv2d(
                #in_channels=512,
                in_channels=2048,
                #in_channels=128,
                out_channels=self.n_classes * len(self.localizations_default),
                kernel_size=(1, int(n_times/(2**5))))
        )
        
        #self.fc_loc = nn.Linear(2 * len(self.localizations_default), 2 * len(self.localizations_default))
        
        #self.fc_clf = nn.Linear(self.n_classes * len(self.localizations_default), self.n_classes * len(self.localizations_default))

    @property
    def is_cuda(self):
        return next(self.parameters())#.is_cuda

    def forward(self, x):
        """forward

        Parameters
        ----------
        x : tensor, shape (n_samples, C, T)
            Input tensor

        Returns:
        --------
        loc : tensor, shape (n_samples, n_default_events * 2)
            Tensor of locations
        clf : tensor, shape (n_samples, n_default_events * n_classes)
            Tensor of probabilities
        """
        batch = x.size(0)
        # x_clone = x.clone()
        # size = x.size()

        # for i in range(5)
        #    x0 = x[:, 0, :]

        # x0= x[:,0,:]
        # x1= x[:,1,:]
        # x2= x[:,2,:]
        # x3= x[:,3,:]
        # x4= x[:,4,:]

        test = sum(np.log2(self.freq_factor))



        tensorss = {}
        to = 0
        '''
        for nr, i in enumerate(self.freq_factor):
            x0 = x[:, to:to+i, :]

            size0 = x0.size()
            x0 = x0.transpose(1, 2).contiguous()
            x0 = x0.view(size0[0], 1, 1, size0[2]*i)
            tensorss[nr] = x0
            to = to + i

        for nr, block in enumerate(self.block1_1):
            tensorss[nr] = block(tensorss[nr])

        for nr, block in enumerate(self.block1_12):
            tensorss[nr] = block(tensorss[nr])

        for nr, block in enumerate(self.block1_13):
            tensorss[nr] = block(tensorss[nr])

        a = list(range( len(self.freq_factor))) * 10


        for nr, block in enumerate(self.blocks):
            tensorss[a[nr]] = block(tensorss[a[nr]])


        #Find indexes

        freq_factor_copy = self.freq_factor.copy()

        for nr, block in enumerate(self.blocks_ekstra):
            index = freq_factor_copy.index(max(freq_factor_copy))
            tensorss[a[index]] = block(tensorss[a[index]])
            freq_factor_copy[index] = freq_factor_copy[index]/2


        z_cat = tensorss[0]

        for nr in range(1,len(self.freq_factor)):
            z_cat = torch.cat((z_cat, tensorss[nr]), 2)

        z_cat = self.dropout_layer(z_cat)
        '''
        z_cat = self.resnet(x)
        z_cat, (hn, cn) = self.rnn(z_cat.squeeze(2).permute(2,0,1))
        z_cat = z_cat.permute(1,2,0)
        z_cat = z_cat.unsqueeze(2)
        
        z_cat = self.dropout_layer(z_cat)
        
        
        loc = self.localization(z_cat).view(batch, -1, 2)
        clf = self.classification(z_cat).view(batch, -1, self.n_classes)
        
        #print('Shape loc1')
        #print(loc1.shape)
        #print('Shape clf1')
        #print(clf1.shape)
        
        #z_cat = torch.flatten(z_cat,1)
        #loc = torch.flatten(loc,1)
        #clf = torch.flatten(clf,1)
        
        
        
        #loc = self.fc_loc(loc).view(batch, -1, 2)
        #clf = self.fc_clf(clf).view(batch, -1, self.n_classes)
        
        #loc = self.localization(z_cat).squeeze().view(batch, -1, 2)
        #clf = self.classification(z_cat).squeeze().view(batch, -1, self.n_classes)
        
        return loc, clf


if __name__ == "__main__":
    n_channels = 4
    n_times = 20 * 128
    n_classes = 2
    model = EventDetector(
        n_times=n_times,
        n_channels=n_channels,
        k_max=8,
        factor_overlap=4,
        n_classes=n_classes).cuda

    x = np.random.randn(10, n_channels, n_times)
    x = Variable(torch.from_numpy(x).float()).cuda
    # x = torch.from_numpy(x).float().cuda()

    z = model(x)
    print(z[0].shape, z[1].shape)
    # print(z.shape)
    # print(z[0].shape, z[1].shape, z[2].shape)
    # print(z[0].shape, z[1].shape, z[2].shape, z[3].shape)
