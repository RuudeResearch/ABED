
"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""


import pandas as pd
import numpy as np
import os

#os.chdir('C:/Users/45223/OneDrive - Danmarks Tekniske Universitet/stanford/Project/copy_of_sherlock/event_detection-master1-sep26')


from datagenerator.base_datagen import DataGen


class DataGen(DataGen):
    """DataGen: generator of data from memmaps
    """

    def __init__(self, df, selected_channels, window, downsampling,
                 minimum_overlap,channels,
                 index_on_events=False, ratio_positive=None, ratio_positive1=None, ratio_positive2=None, ratio_positive3=None,traindata_factor=0.5, normalize=True):

        self.df = df.copy()
        self.selected_channels = selected_channels
        self.number_of_classes = pd.unique(self.df.event)
        self.window = window
        self.downsampling = downsampling
        self.fs = pd.unique(self.df.fs)[0] / self.downsampling
        self.window_size = int(self.window * self.fs)
        self.input_size = self.window_size
        self.ratio_positive = ratio_positive
        self.ratio_positive1 = ratio_positive1
        self.ratio_positive2 = ratio_positive2
        self.ratio_positive3 = ratio_positive3
        self.normalize = normalize
        self.index_on_events = index_on_events
        self.minimum_overlap = minimum_overlap
        self.traindata_factor=traindata_factor 

        self.channels_ = channels
      #  self.channels_ = [
      #   "Flow","Therm",
     #    "Thor", "Abdo", "SpO2", "Snore", "Snor2e" 
    #]
        self.number_of_channels = len(self.channels_)

        super(DataGen, self).__init__()


if __name__ == "__main__":
    
    os.chdir('C:/Users/45223/Desktop/Clean_flow')
    #df = pd.read_csv("data/final_SS2/m_files/info.csv")
    
    #df = pd.read_csv("data/mesa/info_mesa_apr30_lite.csv", delimiter=';')
    df = pd.read_csv("data/wsc_witharousal/info_wsc.csv",delimiter=',')
    
    df["fs"] = df["fs"].values.astype(np.int)
    

    train_gen = DataGen(
        df.iloc[:5],
        selected_channels=["Flow","Therm","Thor", "Abdo", "SpO2", "Snore"], window=20, downsampling=1,
        minimum_overlap=0.5, channels = ["Flow","Therm","Thor", "Abdo", "SpO2", "Snore"],
        index_on_events=True, ratio_positive=0.5, normalize=True)
