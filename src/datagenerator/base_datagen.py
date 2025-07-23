
"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""


from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset

#For filtering 
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.signal import freqz

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


class DataGen(Dataset):
    """DataGen: generator of data from memmaps
    """

    def __init__(self):

        # preload memmaps for signal and events
        self.eegs = {}
        self.events = {}
        self.events_number_of_event = {}
        self.choice ={} 

        print("preloading memmaps")
        for r in tqdm(pd.unique(self.df.record)):
            self.eegs[r] = {}

            eeg_f = pd.unique(self.df[self.df.record == r].eeg_file)[0]
            n_times = pd.unique(self.df[self.df.record == r].n_times)[0]

            
            self.eegs[r]["data"] = np.memmap(
                eeg_f,
                dtype='float32',
                mode='r',
                shape=(self.number_of_channels, n_times))[:, ::self.downsampling]
            
            

            ## normalization parameters burde åbnes
            #self.eegs[r]['means'] = np.mean(
            #    self.eegs[r]["data"], axis=1, keepdims=True)
            #self.eegs[r]['stds'] = np.std(
            #    self.eegs[r]["data"], axis=1, keepdims=True)

            ## normalization parameters for alternative
            #self.eegs[r]['q95'] = np.percentile(
             #   self.eegs[r]["data"],95, axis=1, keepdims=True)

            #self.eegs[r]['q5'] = np.percentile(
             #   self.eegs[r]["data"],5, axis=1, keepdims=True)



            idx_channels = []

            for ic, c in enumerate(self.channels_):

                for c_ in self.selected_channels:
                    if c in c_:
                        idx_channels.append(ic)

            idx_channels = np.asarray(idx_channels).astype(int)

            self.eegs[r]["idx_channels"] = idx_channels
            self.eegs[r]["number_of_windows"] = (
                self.eegs[r]["data"].shape[1] // self.window_size)

            self.events[r] = {}
            event_count = 0
            for event in pd.unique(self.df.event):

                if self.df[
                        (self.df.record == r) &
                        (self.df.event == event)].shape[0] != 0:
                    event_f = self.df[
                        (self.df.record == r) &
                        (self.df.event == event)].event_file.values[0]
                    n_events = self.df[
                        (self.df.record == r) &
                        (self.df.event == event)].n_events.values[0]
                    event_label = self.df[
                        (self.df.record == r) &
                        (self.df.event == event)].label.values[0]

                    if not os.path.isfile(event_f):
                        continue
                    event_count += n_events
                    self.events[r][event] = {}
                    self.events[r][event]["data"] = np.memmap(
                        event_f,
                        dtype='float32',
                        mode='r+', #Was r before
                        shape=(2, n_events)) * self.fs
                    self.events[r][event]["label"] = float(event_label)
            self.events_number_of_event[r] = event_count

        # for each index find correct filename
        self.index_to_record = []


        # for each inde give offset in record
        self.index_to_record_index = []
        if self.index_on_events:
            for record, n_events in self.events_number_of_event.items():
                self.index_to_record.extend([record] * int(n_events *self.traindata_factor )) 
                self.index_to_record_index.extend(range(int(n_events *self.traindata_factor )))  # useless
        else:
            for record, eeg in self.eegs.items():
                self.index_to_record.extend(
                    [record] * (eeg["number_of_windows"]*2-2))
                self.index_to_record_index.extend(
                    range(eeg["number_of_windows"]*2-2))

        # set data extractor:
        if self.ratio_positive in [None, False]:
            self.extract_data = self.get_sample
        else:
            self.extract_data = self.extract_balanced_data

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):
        r = self.index_to_record[idx]
        record_index = self.index_to_record_index[idx]
        eeg, events = self.extract_data(self.eegs[r]["data"],
                                        self.events[r],
                                        record_index)

        #if eeg.shape[0] > 10.5: #delay channels 
        #    eeg[4][:-32*30]=eeg[4][32*30:]
        #    eeg[4][-32*30:]=eeg[4][-32*30:]*0

        #    eeg[5][:-32*10]=eeg[5][32*10:]
        #    eeg[5][-32*10:]=eeg[5][-32*10:]*0

       # # normalization step
        if self.normalize:
            #means = self.eegs[r]["means"] #for hele data
            #stds = self.eegs[r]["stds"]


            means = np.mean(
                self.eegs[r]["data"], axis=1, keepdims=True)
            stds = np.std(
                self.eegs[r]["data"], axis=1, keepdims=True)
            chan = self.channels_
            eeg, events = NormalizeRecord(means, stds, chan)(eeg, events)

        #if self.normalize:
         #   q95 = self.eegs[r]["q95"]
          #  q5 = self.eegs[r]["q5"]
           # chan = self.channels_
            #eeg, events = NormalizeRecord(q95, q5, chan)(eeg, events)

        idx_channels = self.eegs[r]["idx_channels"]

        #if 4 in idx_channels:   #For finding difference in O2
        #    eeg[4] = eeg[4][32*30]-eeg[4]
        #    eeg[4][-32*30:-1] = eeg[4][-32*30:-1]*0


        eeg_ = eeg[idx_channels]

        # return eeg_, events
        return torch.FloatTensor(eeg_), torch.FloatTensor(events)

    

    def extract_balanced_data(self, eeg, events, index=None):
        """ Extract a peculiar index or random
        """
        num_CSA =0
        num_HYPO = 0
        num_OSA = 0
        num_NAD_HYPO = 0

        eeg_data, events_data = self.get_sample(eeg, events, index=None)

        if {'HYPO'}.issubset(events):
            
            #if events["HYPO"]["data"][1, -1] > len(eeg):
            num_HYPO = len(events["HYPO"]["data"][0, :])
            
        if {'NAD'}.issubset(events):
            num_NAD_HYPO = len(events["NAD"]["data"][0, :])


        if {'OSA'}.issubset(events):
            #if events["OSA"]["data"][1, -1] > len(eeg):
            num_OSA = len(events["OSA"]["data"][0, :])

        if {'CSA'}.issubset(events):
            #if events["CSA"]["data"][1, -1] > len(eeg):
            num_CSA = len(events["CSA"]["data"][0, :])

        #(num_OSA*self.ratio_positive + num_CSA*self.ratio_positive1 + num_HYPO*self.ratio_positive2)/1
        '''
        p1 = [ (num_OSA*self.ratio_positive + num_CSA*self.ratio_positive1 + num_HYPO*self.ratio_positive2 + num_NAD_HYPO*self.ratio_positive3)/1.5 , self.ratio_positive * num_OSA,
             self.ratio_positive1 * num_CSA, self.ratio_positive2 * num_HYPO, self.ratio_positive3 * num_NAD_HYPO]
        '''
        p1 = [ (num_OSA*self.ratio_positive + num_CSA*self.ratio_positive1 + num_HYPO*self.ratio_positive2 + num_NAD_HYPO*self.ratio_positive3)/1.5 , self.ratio_positive * num_OSA,
             self.ratio_positive1 * num_CSA, self.ratio_positive2 * num_HYPO, self.ratio_positive3 * num_NAD_HYPO]
        
        p2 = p1/np.sum(p1)
        
        #print(print(p2))
        
        choice = np.random.choice(
            [0, 1, 2, 3, 4], p=p2)  

        if choice == 0:
            while len(events_data) > 0:
                eeg_data, events_data = self.get_sample(
                    eeg, events, index=None)
        elif choice == 1:
            random_event = np.random.randint(num_OSA)
            start = events["OSA"]["data"][0, random_event]
            end = events["OSA"]["data"][0, random_event] + self.window_size - events["OSA"]["data"][1, random_event]

            if start < end:
                index = int(np.random.randint(start, end) / self.window_size)
            else: 
                index = int(np.random.randint(start, start+self.window_size*0.2) / self.window_size)
            eeg_data, events_data = self.get_sample(eeg, events, index,training=True)

        elif choice == 2:
            random_event = np.random.randint(num_CSA)
            start= events["CSA"]["data"][0, random_event]
            end = events["CSA"]["data"][0, random_event]+self.window_size-events["CSA"]["data"][1, random_event]
            if start < end:
                index = int(np.random.randint(start, end) / self.window_size)
            else: 
                index = int(np.random.randint(start, start+self.window_size*0.2) / self.window_size)
            eeg_data, events_data = self.get_sample(eeg, events, index,training=True)

        elif choice == 3:
            random_event = np.random.randint(num_HYPO)
            start = events["HYPO"]["data"][0, random_event]
            end = events["HYPO"]["data"][0, random_event] + self.window_size - events["HYPO"]["data"][1, random_event]
            if start < end:
                index = int(np.random.randint(start, end) / self.window_size)
            else: 
                index = int(np.random.randint(start, start+self.window_size*0.2) / self.window_size)
            eeg_data, events_data = self.get_sample(eeg, events, index,training=True)
        elif choice == 4:
            random_event = np.random.randint(num_NAD_HYPO)
            start = events["NAD"]["data"][0, random_event]
            end = events["NAD"]["data"][0, random_event] + self.window_size - events["NAD"]["data"][1, random_event]
            if start < end:
                index = int(np.random.randint(start, end) / self.window_size)
            else: 
                index = int(np.random.randint(start, start+self.window_size*0.2) / self.window_size)
            eeg_data, events_data = self.get_sample(eeg, events, index,training=True)
        return eeg_data, events_data

    def get_sample(self, eeg, events, index=None,training=False):
        if index is None:
            
            index = np.random.randint(eeg.shape[1] - self.window_size)

        elif len(eeg[0, index*self.window_size:index*self.window_size + self.window_size])==len(eeg[0, 10*self.window_size:10*self.window_size + self.window_size]) and training is True and index>0.5: #test if it is the end
            index = index * self.window_size

        elif training is False:
            index = int(index * self.window_size*0.5)


        else:
             index = np.random.randint(eeg.shape[1] - self.window_size)

        eeg_data = np.vstack(eeg[:, index:index + self.window_size])



        events_data = []
        for event_name, event in events.items():
            starts, durations = event["data"][0, :], event["data"][1, :]
            # Relative start stop
            starts_relative = (starts - index) / self.window_size
            durations_relative = durations / self.window_size
            stops_relative = starts_relative + durations_relative

            # Find valid start or stop
            valid_starts_index = np.where(
                (starts_relative > 0) * (starts_relative < 1))[0]
            valid_stops_index = np.where(
                (stops_relative > 0) * (stops_relative < 1))[0]

            # merge them
            valid_indexes = set(
                list(valid_starts_index) + list(valid_stops_index))

            # Annotations contains valid index with minimum overlap requirement
            for valid_index in valid_indexes:
                if (valid_index in valid_starts_index) and (valid_index in valid_stops_index):
                    events_data.append((float(starts_relative[valid_index]),
                                        float(stops_relative[valid_index]), event["label"]))
                elif valid_index in valid_starts_index:
                    if ((1 - starts_relative[valid_index]) /
                            durations_relative[valid_index]) > self.minimum_overlap:
                        events_data.append((float(starts_relative[valid_index]), 1, event["label"]))

                elif valid_index in valid_stops_index:
                    if ((stops_relative[valid_index]) / durations_relative[valid_index]) > self.minimum_overlap:
                        events_data.append((0, float(stops_relative[valid_index]), event["label"]))

        return eeg_data, events_data


#class NormalizeRecord:

    #def __init__(self, means, stds):
     #   self.means = means
    #    self.stds = stds
   #     self.chan

  #  def __call__(self, eeg, events):
 #       eeg = (eeg - self.means) / self.stds

#        return eeg, events

class NormalizeRecord:

    def __init__(self, means, stds, chan):
        self.means = means
        self.stds = stds
        self.chan = chan

    def __call__(self, eeg, events):





       

        #eeg1 = (eeg - self.means) / self.stds
        eeg1 = eeg
        #x = eeg1
        #eeg1 = butter_bandpass_filter(x, 0.1, 0.11, 32, order=6)
 
 

        #filter1 = np.ones(32)
        
        #Filter skal gøres inden
        #for nr, chan_name in enumerate(self.chan):
        #    if chan_name == "SpO2":
        #        eeg1[nr, :] = np.convolve(
        #            eeg[nr, :], filter1, mode='same')/(32*100)
        #    elif chan_name == "Snore":
        #        eeg1[nr,:] = eeg[nr, :]/self.stds[nr]
        #    elif chan_name == "Flow":
        #        x = eeg1[nr, :]
        #        eeg1[nr,:] = butter_bandpass_filter(x, 0.1, 10, 32, order=4)
        #    elif chan_name == "Therm":
        #        x = eeg1[nr, :]
        #        eeg1[nr,:] = butter_bandpass_filter(x, 0.1, 10, 32, order=4)
        #    elif chan_name == "Thor":
        #        x = eeg1[nr, :]
        #        eeg1[nr,:] = butter_bandpass_filter(x, 0.1, 10, 32, order=4)
        #    elif chan_name == "Abdo":
        #        x = eeg1[nr, :]
        #        eeg1[nr,:] = butter_bandpass_filter(x, 0.1, 10, 32, order=4)   





        return eeg1, events        

#class NormalizeRecord:

 #   def __init__(self, q95, q5, chan):
  #      self.q95 = q95
   #     self.q5 = q5
   

    #def __call__(self, eeg, events):

        #eeg = (eeg - self.means) / self.stds
     #   eeg1 = 2*(eeg - self.q5)/( self.q95 - self.q5)-1

        # normalization parameters for alternative
      #  eegs_short_95 = np.percentile(
       #     eeg, 95, axis=1, keepdims=False)

        #eegs_short_5 = np.percentile(
         #   eeg, 5, axis=1, keepdims=False)

      #  filter1 = np.ones(32)

        #for nr, chan_name in enumerate(self.chan):
         #   if chan_name == "Flow":
          #      eeg1[nr,:] = 2*(eeg[nr,:] - eegs_short_5[nr])/( eegs_short_95[nr] - eegs_short_5[nr])-1
           # elif chan_name == "Abdo":
            #    eeg1[nr, :] = 2 * (eeg[nr, :] - eegs_short_5[nr]) / (eegs_short_95[nr] - eegs_short_5[nr]) - 1
            #elif chan_name == "Thor":
             #   eeg1[nr, :] = 2 * (eeg[nr, :] - eegs_short_5[nr]) / (eegs_short_95[nr] - eegs_short_5[nr]) - 1
            #elif chan_name == "SpO2":
             #   eeg1[nr, :] = np.convolve(
              #      eeg[nr, :], filter1, mode='same')/(32*100)
            #elif chan_name == "Aux_AC":
             #   eeg1[nr, :] = 2 * (eeg[nr, :] - eegs_short_5[nr]) / (eegs_short_95[nr] - eegs_short_5[nr]) - 1

    #    return eeg1, events
