# -*- coding: utf-8 -*-
"""
@author: Magnus Ruud Kjær
"""

# general imports
import pandas as pd
import os
import numpy as np
import scipy.io
from tqdm import tqdm
from scipy.io import savemat

# sklearn imports
from sklearn.model_selection import train_test_split

# torch imports
import torch

# changeing directory in order to access self made classes
#os.chdir('C:/Users/45223/OneDrive - Danmarks Tekniske Universitet/stanford/Project/copy_of_sherlock/event_detection-master1-sep26')
#os.chdir('C:/Users/45223/Desktop/Clean_flow')

# detection imports
from datagenerator.datagen import DataGen
#from event_detection.datasets.mesa.m_loader import DataGen
from model.event_detector import EventDetector
from utilities.paper_utils import give_storing_paths
from metrics.metrics import evaluation
from metrics.metrics import multi_evaluation

#sweep imports
#import wandb

file_name = __file__

if __name__ == "__main__":

    #loggin onto wandb for sweeping
    #wandb.login()
    
    #defining sweep using 
    '''
    sweep_config = {
        'method': 'random'
        }
    
    metric = {
        'name': 'loss',
        'goal': 'minimize'
        }
    sweep_config['metric'] = metric
    
    parameters_dict = {
    'windue': {
        'values': [120, 240]
        },
    'lagg': {
        'values': [7, 11, 15]
        },
    'droop': {
          'values': [0, 0.2, 0.4]
        },
    }
    sweep_config['parameters'] = parameters_dict
    '''
    

    #fixing random generation for reproduceability
    np.random.seed(42)
    torch.manual_seed(42)
    
    #print(file_name)
    print(__file__)
    #print(os.getcwd())
    
    '''
    scores_path, histories_path, weights_path, pred_path = give_storing_paths(
        __file__, os.getcwd())
    '''
    scores_path = '/home/users/magnusrk/Clean_flow_NAD_HYPO/scores/main.csv'
    histories_path = '/home/users/magnusrk/Clean_flow_NAD_HYPO/histories/main.csv'
    weights_path = '/home/users/magnusrk/Clean_flow_NAD_HYPO/weights/main'
    pred_path = '/home/users/magnusrk/Clean_flow_NAD_HYPO/predictions/main'
    
    # data loader parameters
    channels = ["Flow","Therm","Thor", "Abdo", "SpO2", "Arousal", "Wake"]#, "Snore"]
    start_name2 = 'Rebuttal_model_0point025hz_'

    #defining hyper parameters
    #observation if run locally and lr set to 0.005 and dropout at 0.7 the initial training losses are very high
    drooop = 0.9 #dropout used
    windue = 240#120 #window width in seconds
    leann = 0.001 # learning rate
    lagg = 5 #layers in the CNN
    nrr = 30
    linearlayer = 0
    maxpool = 2
    batchsize= 128 #18 is default
    n_epoch = 8#8
    patience_val = 100
    lr_decrease_patience_val = 30
    resnet_architecture = [3, 4, 6, 3]
    #resnet_architecture = [2, 2, 2, 2]
    

    window = windue
    downsampling = 1
    minimum_overlap = 0.5
    selected_channels =["Flow","Therm","Thor", "Abdo", "SpO2", "Arousal", "Wake"]
    freq_factor = [1, 1, 1, 1, 1, 1, 1]

    #detection_threshold = [0.8, 0.7, 0.6, 0.5]
    #detection_threshold = [0.8]
    detection_threshold =[0.91,0.81,0.71,0.61,0.51]
    #detection_threshold = [0.35, 0.4, 0.25, 0.3]
    #detection_threshold = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    #detection_threshold = [0.5]   
    
    multicoohord=0
    
    
    '''
    df_wsc_arousal = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/wsc_mm_all2/info_wsc_num_record.csv",delimiter=',')
    df_mros_NAD = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/mros_mm_NAD_HYPO/info_mros.csv",delimiter=',')
    df_mesa_NAD = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/mesa_mm_NAD_HYPO/info_mesa.csv",delimiter=',')
    df_cfs = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/cfs_mm_all/info_cfs.csv",delimiter=',')
    df_shhs = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/shhs_mm_w0_switch_o2fix/info_mesa.csv",delimiter=',')
    df_apoe = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/apoe_mm_annotation_fix/info_apoe.csv",delimiter=',')
    df_chat = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/chat_mm_all/info_chat.csv",delimiter=',')


     
    df_wsc_arousal = df_wsc_arousal[np.isin(df_wsc_arousal.record.values, df_wsc_arousal.record.unique()[:])]
    df_mros_NAD = df_mros_NAD[np.isin(df_mros_NAD.record.values, df_mros_NAD.record.unique()[:])]
    df_mesa_NAD = df_mesa_NAD[np.isin(df_mesa_NAD.record.values, df_mesa_NAD.record.unique()[:])]
    df_cfs = df_cfs[np.isin(df_cfs.record.values, df_cfs.record.unique()[:])]
    df_shhs = df_shhs[np.isin(df_shhs.record.values, df_shhs.record.unique()[:])]
    df_apoe = df_apoe[np.isin(df_apoe.record.values, df_apoe.record.unique()[:])]
    df_chat = df_chat[np.isin(df_chat.record.values, df_chat.record.unique()[:])]
    df = pd.concat([df_mesa_NAD,df_wsc_arousal,df_mros_NAD,df_cfs,df_apoe,df_chat], ignore_index=True, sort=False)

    df_test = pd.concat([df_mesa_NAD[:250],df_wsc_arousal[:250],df_mros_NAD[:500],df_cfs[:100],df_apoe[:100],df_chat[:50]], ignore_index=True, sort=False)
    '''
    
    # df_wsc_arousal = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/wsc_mm_all2/info_wsc_num_record.csv",delimiter=',')
    # df_mros_NAD = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/mros_mm_NAD/info_mros.csv",delimiter=',')
    # df_mesa_NAD = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/mesa_mm_NAD/info_mesa.csv",delimiter=',')
    # df_cfs = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/cfs_mm_all/info_cfs.csv",delimiter=',')
    # df_shhs = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/shhs_mm_w0_NAD/info_shhs.csv",delimiter=',')

    df_wsc_arousal = pd.read_csv("/scratch/users/magnusrk/ABED/wsc_mm_all2/info_wsc_num_record.csv",delimiter=',')
    df_mros_NAD = pd.read_csv("/scratch/users/magnusrk/ABED/mros_mm_NAD/info_mros.csv",delimiter=',')
    df_mesa_NAD = pd.read_csv("/scratch/users/magnusrk/ABED/mesa_mm_NAD/info_mesa_ctc.csv",delimiter=',')
    df_cfs = pd.read_csv("/scratch/users/magnusrk/ABED/cfs_mm_all/info_cfs.csv",delimiter=',')
    # df_shhs = pd.read_csv("/scratch/users/magnusrk/Clean_flow_arousal/data/shhs_mm_w0_NAD/info_shhs.csv",delimiter=',')



     
    df_wsc_arousal = df_wsc_arousal[np.isin(df_wsc_arousal.record.values, df_wsc_arousal.record.unique()[:])]
    df_mros_NAD = df_mros_NAD[np.isin(df_mros_NAD.record.values, df_mros_NAD.record.unique()[:])]
    df_mesa_NAD = df_mesa_NAD[np.isin(df_mesa_NAD.record.values, df_mesa_NAD.record.unique()[:])]
    df_cfs = df_cfs[np.isin(df_cfs.record.values, df_cfs.record.unique()[:])]
    # df_shhs = df_shhs[np.isin(df_shhs.record.values, df_shhs.record.unique()[:])]


    #df = pd.concat([df_mesa_NAD,df_wsc_arousal,df_mros_NAD,df_cfs,df_shhs,df_apoe], ignore_index=True, sort=False)
    
    
    # df = pd.concat([df_mesa_NAD,df_wsc_arousal,df_mros_NAD,df_cfs,df_shhs], ignore_index=True, sort=False)

    # df_test = pd.concat([df_mesa_NAD[:250],df_wsc_arousal[:250],df_mros_NAD[:500],df_cfs[:100],df_shhs[:1200]], ignore_index=True, sort=False)
    
    df = pd.concat([df_mesa_NAD,df_wsc_arousal,df_mros_NAD,df_cfs], ignore_index=True, sort=False)

    df_test = pd.concat([df_mesa_NAD[:250],df_wsc_arousal[:250],df_mros_NAD[:500],df_cfs[:100]], ignore_index=True, sort=False)


    # df = pd.concat([df_wsc_arousal,df_cfs], ignore_index=True, sort=False)

    # df_test = pd.concat([df_wsc_arousal[:250],df_cfs[:100]], ignore_index=True, sort=False)

    
    #df = df_shhs

    #df_test = df_shhs[:1190]
    
    
    #df = df_mesa_NAD
    #df_test = df_mesa_NAD[:10]
    
    #df = df[df.event != 'NAD-HYPO']
    
    
    
    
    df = df[df.record!=3347] #id 3347 is weird CSA and HYPO of an hour length 
    df = df[df.record!=718] #only data from first hour, and event afterwards

    df = df[df.record!=1004714]
    df = df[df.record!=1005014]#1006852, 1005316, 1005585, 1006119
    df = df[df.record!=1005013]
    df = df[df.record!=800695]
    df = df[df.record!=800696]
    df = df[df.record!=800699]
    df = df[df.record!=800698]
    df = df[df.record!=800701]
    df = df[df.record!=800753]

    df = df[df.record!=205686] # look at shhs file, this is messed up....

    
    
    df_test = df_test[df_test.record!=3347] #id 3347 is weird CSA and HYPO of an hour length 
    df_test = df_test[df_test.record!=718] #only data from first hour, and event afterwards

    df_test = df_test[df_test.record!=1004714]
    df_test = df_test[df_test.record!=1005014]#1006852, 1005316, 1005585, 1006119
    df_test = df_test[df_test.record!=1005013]
    df_test = df_test[df_test.record!=800695]
    df_test = df_test[df_test.record!=800696]
    df_test = df_test[df_test.record!=800699]
    df_test = df_test[df_test.record!=800698]
    df_test = df_test[df_test.record!=800701]
    df_test = df_test[df_test.record!=800753]

    df_test = df_test[df_test.record!=205686]

    
    
    
    
    df = df[np.isin(df.record.values, df.record.unique()[:])]
    
    
    '''
    
    df_preAPOE = df_preAPOE[df_preAPOE.record!=3347] #id 3347 is weird CSA and HYPO of an hour length 
    df_preAPOE = df_preAPOE[df_preAPOE.record!=718] #only data from first hour, and event afterwards

    df_preAPOE = df_preAPOE[df_preAPOE.record!=1004714]
    df_preAPOE = df_preAPOE[df_preAPOE.record!=1005014]#1006852, 1005316, 1005585, 1006119
    df_preAPOE = df_preAPOE[df_preAPOE.record!=1005013]
    
    df = pd.concat([df_preAPOE,df_APOE], ignore_index=True, sort=False) 
    '''
    #df = df[np.isin(df.record.values, df.record.unique()[:124])]#amount of data #max 40 on local
    df = df[np.isin(df.record.values, df.record.unique()[:])]#using ALL
    sfreq = pd.unique(df["fs"])[0] // downsampling

    # records
    records = sorted(pd.unique(df.record)) #used sorted before 

    #for visu
    gen_params = dict(
        selected_channels=selected_channels,
        window=window,
        downsampling=downsampling,
        minimum_overlap=minimum_overlap,
        channels=channels)

    for idx_split in [nrr]:

        if multicoohord==1:
            df_test_pick = df_multi_arousal
            #print(df_test_pick)
            r_test = sorted(pd.unique(df_test_pick.record))
            #print(df[df.record == r_test[0]])
        else:
            df_tester = df_test
            r_test = sorted(pd.unique(df_tester.record)) # one third of data
            #r_test = records[:1000] #MESA 1000:
            #r_test = records[:1000] #SHHS 1000:

        r_ = sorted([r for r in records if r not in r_test])
        print(r_)
        r_train, r_val = train_test_split(
            r_, train_size=0.9,
            test_size=0.1,
            random_state=idx_split)
        df_save_r_val = pd.DataFrame([r_val])
        df_save_r_val.to_csv('/home/users/magnusrk/Clean_flow_NAD_HYPO/validation_list.csv')
        
        print("Training on {} records".format(len(r_train)))
        print("Validation on {} records".format(len(r_val)))
        print("Prediction on {} records".format(len(r_test))) #was r_test

        df_train = df[df.record.isin(r_train)].reset_index()
        df_val = df[df.record.isin(r_val)].reset_index()

        #r_test = r_val #For hyperparameter
        
        #Calculate boost

        n_HYPO = df[df['label'] == 2]['n_events'].sum()
        factor_HYPO = 1 
        
        n_NAD_HYPO = df[df['label'] == 3]['n_events'].sum()
        factor_NAD_HYPO = n_HYPO/n_NAD_HYPO

        n_OSA = df[df['label']==0]['n_events'].sum()
        factor_OSA=n_HYPO/n_OSA
        #factor_OSA=1  

        n_CSA = df[df['label']==1]['n_events'].sum()
        factor_CSA=(n_HYPO/n_CSA) 
        #factor_CSA = 1

        train_gen = DataGen(
            df_train, selected_channels, window, downsampling,
            minimum_overlap,channels,
            index_on_events=True, ratio_positive=factor_OSA, ratio_positive1=factor_CSA, ratio_positive2=factor_HYPO, ratio_positive3=factor_NAD_HYPO,traindata_factor=1) #was 0.5 3 0.3

        val_gen = DataGen(
            df_val, selected_channels, window, downsampling,
            minimum_overlap,channels,
            index_on_events=True, ratio_positive=factor_OSA, ratio_positive1=factor_CSA, ratio_positive2=factor_HYPO, ratio_positive3=factor_NAD_HYPO,traindata_factor=1) #was 0.5 3 0.3

        
        for duration in [10]: 

            lr=leann#1e-3
            epochs=n_epoch#5#1
            dropout=drooop
            n_channels=len(selected_channels)
            partial_eval=-1#5000#5000#5000#3500#3500 #6500#750#-1#10000#7700#5501#7700#7700#7700#7000#7700#7700    #-1 for non

            model = EventDetector(
                n_channels=n_channels,n_classes=4, n_freq_channels = 0,freq_factor =freq_factor,
                n_times=(window * sfreq), num_workers=10, fs=sfreq,
                histories_path="{}_{}_{}.csv".format(
                    histories_path.split(".")[0], idx_split, duration),
                weights_path="{}_{}_{}.pth".format(
                    weights_path, idx_split, duration),
                loss="worst_negative_mining",
                default_event_sizes=[duration * sfreq,duration * sfreq*2,duration * sfreq*3,duration * sfreq*4,duration * sfreq*5,duration * sfreq*6,duration * sfreq*7,duration * sfreq*8,duration * sfreq*9,duration * sfreq*10,duration * sfreq*11,duration * sfreq*12],
                factor_overlap=2,
                lr=lr,
                patience=patience_val,
                lr_decrease_patience=lr_decrease_patience_val,
                epochs=epochs,
                k_max=lagg,max_pooling=maxpool, batch_size=batchsize,dropout=dropout  #was 9
                ,partial_eval=partial_eval,
                linearlayer=linearlayer,
                RES_architecture=resnet_architecture                
            )
            #model.summary()
            valloss = model.fit_generator(train_gen, val_gen)

            for op_AHI in range(1):

                # hp selection
                if op_AHI==1:
                    columns0 = ["op_ahi_0", "clf_th"]
                    columns1 = ["op_ahi_1", "clf_th"]
                    columns2 = ["op_ahi_2", "clf_th"]
                    columns3 = ["op_ahi_3", "clf_th"]
                else:
                    columns0 = ["f1_0", "clf_th"]
                    columns1 = ["f1_1", "clf_th"]
                    columns2 = ["f1_2", "clf_th"]
                    columns3 = ["f1_3", "clf_th"]

                scores_val0 = dict()
                scores_val1 = dict()
                scores_val2 = dict()
                scores_val3 = dict()

                for idx_r, r in enumerate(tqdm(list(r_val))): 
                    pass

                    if df[df.record == r].n_events.values[0] != 0:
                        pred_gen = DataGen(
                            df[df.record == r], selected_channels,
                            window, downsampling,
                            minimum_overlap,channels,
                            index_on_events=False, ratio_positive=None)

                        y_true, y_pred, y_prob = model.predict_generator(
                            pred_gen, detection_threshold)
                            
                        #y_pred[0,1,:]=y_pred[1,1,:]
                        #y_pred[0,2,:]=y_pred[2,2,:]
                        #y_pred[0,3,:]=y_pred[3,3,:]
                        
                        
                        #y_pred[0,0,:] = y_pred[0,0,:] - np.multiply(y_pred[0,0,:],y_pred[0,1,:])
                        #y_pred[0,3,:] = y_pred[0,3,:] - np.multiply(y_pred[0,3,:],y_pred[0,1,:])- np.multiply(y_pred[0,3,:],y_pred[0,0,:])
                        #y_pred[0,2,:]= y_pred[0,2,:] - np.multiply(y_pred[0,2,:], y_pred[0,1,:])- np.multiply(y_pred[0,2,:], y_pred[0,0,:]) - np.multiply(y_pred[0,2,:],y_pred[0,3,:])
                        
    
                        #y_pred = y_pred[0,:,:]

                        s0_ = []
                        s1_ = []
                        s2_ = []
                        s3_ = []

                        for idx_thres, thres in enumerate(detection_threshold):
                            print(thres)

                            #y_true=y_true[0,:] #ny
                            #y_pred=y_pred[:,0,:]

                            s = multi_evaluation(
                                y_true.squeeze(), y_pred[idx_thres].squeeze(),
                                sfreq=pred_gen.fs, iou_ths=[0.001])
                            s["clf_th"] = thres
                            s0_.append(s[columns0])
                            s1_.append(s[columns1])
                            s2_.append(s[columns2])
                            s3_.append(s[columns3])


                        if op_AHI==1:
                            scores_val0[r] = pd.concat(s0_).op_ahi_0.values
                            scores_val1[r] = pd.concat(s1_).op_ahi_1.values
                            scores_val2[r] = pd.concat(s2_).op_ahi_2.values
                            scores_val3[r] = pd.concat(s3_).op_ahi_3.values
                        else:
                            scores_val0[r] = pd.concat(s0_).f1_0.values
                            scores_val1[r] = pd.concat(s1_).f1_1.values
                            scores_val2[r] = pd.concat(s2_).f1_2.values
                            scores_val3[r] = pd.concat(s3_).f1_3.values


                scores_val0 = pd.DataFrame(scores_val0)
                scores_val0 = scores_val0.fillna(0)
                scores_val0 = scores_val0.mean(axis=1)
                thres0 = detection_threshold[scores_val0.idxmax()]

                scores_val1 = pd.DataFrame(scores_val1)
                scores_val1 = scores_val1.fillna(0)
                scores_val1 = scores_val1.mean(axis=1)
                thres1 = detection_threshold[scores_val1.idxmax()]

                scores_val2 = pd.DataFrame(scores_val2)
                scores_val2 = scores_val2.fillna(0)
                scores_val2 = scores_val2.mean(axis=1)
                thres2 = detection_threshold[scores_val2.idxmax()]
                
                scores_val3 = pd.DataFrame(scores_val3)
                scores_val3 = scores_val3.fillna(0)
                scores_val3 = scores_val3.mean(axis=1)
                thres3 = detection_threshold[scores_val3.idxmax()]

                if op_AHI == 1:
                    thres0 = detection_threshold[scores_val0.idxmin()]
                    thres1 = detection_threshold[scores_val1.idxmin()]
                    thres2 = detection_threshold[scores_val2.idxmin()]
                    thres3 = detection_threshold[scores_val3.idxmin()]


                

                #for testset in [0,1,2,3,4,5,12,13,14,15,16,17]:#[0]:# [0,1,2,3,4,5,6,7]:# [0,1, 2, 3, 4, 5]:
                if multicoohord == 1:
                    testrange = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
                if multicoohord == 0:
                    testrange = [0]
                
                for testset in testrange:
                #for testset in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]:
                    # prediction
                    scores_test0 = dict()
                    scores_test1 = dict()
                    scores_test2 = dict()
                    scores_test3 = dict()
                    scores_test4 = dict()
                    scores_test5 = dict()
                    scores_test51 = dict()
                    scores_test511 = dict()
                    scores_test6 = dict()
                    scores_test7 = dict()
                    scores_test8 = dict()
                    scores_test9 = dict()
                    scores_test10 = dict()
                    scores_test11 = dict()
                    scores_test111 = dict()
                    scores_test1111 = dict()
                    scores_test12 = dict()
                    scores_test13 = dict()
                    scores_test14 = dict()
                    scores_test15 = dict()
                    scores_test16 = dict()
                    scores_test17 = dict()
                    
                    scores_test312 = dict()
                    scores_test313 = dict()
                    scores_test314 = dict()
                    scores_test315 = dict()
                    scores_test316 = dict()
                    scores_test317 = dict()

                    scores_testsamlet = dict()
                    scores_test0samlet = dict()
                    scores_test1samlet = dict()
                    scores_test2samlet = dict()
                    scores_test3samlet = dict()

                    scores_test171 = dict()
                    scores_test1711 = dict()
                    
                    scores_test3171 = dict()
                    scores_test31711 = dict()
                    
                    scores_test18 = dict()
                    scores_test19 = dict()
                    scores_test20 = dict()
                    scores_test21 = dict()
                    scores_test22 = dict()
                    scores_test23 = dict()
                    scores_test24 = dict()
                    scores_test25 = dict()

                    scores_testsamlet = dict()
                    scores_predsamlet = dict()
                    scores_truesamlet = dict()
                    scores_test0samlet = dict()
                    scores_test1samlet = dict()
                    scores_test2samlet = dict()
                    
                    
                    scores_test319 = dict()
                    scores_test321 = dict()
                    scores_test323 = dict()
                    
                    scores_test3322 = dict()
                    scores_test3323 = dict()
                    scores_test3324 = dict()
                    #min_thres = min(thres0,thres1,thres2)


                    for idx_r, r in enumerate(tqdm(r_test)):
                        print(df[df.record == r])
                        print(selected_channels)
                        print(window)
                        print(downsampling)
                        print(minimum_overlap)
                        print(channels)

                        pred_gen = DataGen(
                            df[df.record == r], selected_channels,
                            window, downsampling,
                            minimum_overlap,channels,
                            index_on_events=False, ratio_positive=None)



                        y_true, y_pred, y_prob = model.predict_generator(
                        pred_gen, detection_threshold=[thres0,thres1,thres2,thres3]) #was -0.1 -0.05 -0.05
                        print('these are the thresholds')
                        print([thres0,thres1,thres2,thres3])

                        #y_true, y_pred1 = model.predict_generator(
                        #pred_gen, detection_threshold=[thres1-0.05]) 

                        #y_true, y_pred2 = model.predict_generator(
                        #pred_gen, detection_threshold=[thres2-0.05])

                        #y_pred[:,1,:]=y_pred1[:,1,:]
                        #y_pred[:,2,:]=y_pred2[:,2,:]

                        y_pred[0,1,:]=y_pred[1,1,:]
                        y_pred[0,2,:]=y_pred[2,2,:]
                        y_pred[0,3,:]=y_pred[3,3,:]
                        
                        
                        y_pred[0,0,:] = y_pred[0,0,:] - np.multiply(y_pred[0,0,:],y_pred[0,1,:])
                        y_pred[0,3,:] = y_pred[0,3,:] - np.multiply(y_pred[0,3,:],y_pred[0,1,:])- np.multiply(y_pred[0,3,:],y_pred[0,0,:])
                        y_pred[0,2,:]= y_pred[0,2,:] - np.multiply(y_pred[0,2,:], y_pred[0,1,:])- np.multiply(y_pred[0,2,:], y_pred[0,0,:]) - np.multiply(y_pred[0,2,:],y_pred[0,3,:])
                        
    
                        y_pred = y_pred[0,:,:]

                        tech_name = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']

                        start_name = '_'
                        if multicoohord == 1:
                            if 10071 > r > 10000: #Eillen's multi scored
                                specific_idx = r-10001
                                #length = [1255680, 1289280, 1230720, 1059840, 1085760, 1247040, 1028160, 1292160, 1038720, 1093440, 1224000, 965760, 1314240, 1264320, 1101120, 1070400, 1180800, 1351680, 1005120 ,1122240 ,1138560, 1083840, 1098240, 998400, 1257600, 1080960, 1032000, 1176000, 1140480, 1127040, 1097280, 1235520, 1181760, 1082880, 964800, 1185600, 1104960, 1273920, 1171200, 1061760, 1119360, 1128960, 1268160, 1297920, 948480, 971520, 1130880, 1351680, 1142400, 1222080, 1265280, 1015680, 1134720, 1262400, 1085760, 1245120, 1209600, 1198080, 1048320, 1194240, 1264320, 1704960, 1233600, 1229760, 1060800, 1105920, 1194240, 1113600, 988800, 1123200]
                                length = [1255680, 1289280, 1230720, 1059840, 1085760, 1247040, 1028160, 1292160, 1038720, 1093440, 1224000, 965760, 1314240, 1264320, 1101120, 1070400, 1180800, 1351680, 1005120 ,1122240 ,1138560, 1083840, 1098240, 998400, 1257600, 1080960, 1032000, 1176000, 1140480, 1127040, 1097280, 1235520, 1181760, 1082880, 964800, 1185600, 1104960, 1273920, 1171200, 1061760, 1119360, 1128960, 1268160, 1297920, 948480, 971520, 1130880, 1351680, 1142400, 1222080, 1265280, 1015680, 1134720, 1262400, 1085760, 1245120, 1209600, 1198080, 1048320, 1194240, 1264320, 1704960, 1233600, 1229760, 1060800, 1105920, 1194240, 1113600, 988800, 1123200]
                                
                                #length = [int(x / 4) for x in length]
                                
                                #num 62 has actually the length 
                                cons_folder1 = "/scratch/users/magnusrk/Clean_flow/data/consensus_em_feb5/weigths1/min_3/"#Magnus path
                                #cons_folder1 = "/scratch/users/jthybo/consensus_scores/consensus_em_feb5/weigths1/min_3/"
                                cons_folder0 = "/scratch/users/jthybo/consensus_scores/consensus_em_feb5/weigths0/min_3/"
                                #cons_folder_maj = "/scratch/users/jthybo/consensus_scores/multiscore_agree_feb5/min_1/"
                                #cons_folder_all = "/scratch/users/jthybo/consensus_scores/consensus_em_alle_nov14/min_2/"
                                #start_name = 'em_feb1_ini3_weigths0_'
                                tech_name = ['hs','hsc','lr','mjp','tech1','tech3','hs','hsc','lr','mjp','tech1','tech3','hs','hsc','lr','mjp','tech1','tech3','hs','hsc','lr','mjp','tech1','tech3','hs','hsc','lr','mjp','tech1','tech3','hs','hsc','lr','mjp','tech1','tech3','hs','hsc','lr','mjp','tech1','tech3','hs','hsc','lr','mjp','tech1','tech3','hs','hsc','lr','mjp','tech1','tech3']

                              
                            
                                
                                if testset < 5.5:
                                    start_name = 'em_feb5_ini3_weigths1_'

                                    if specific_idx < 34.5:
                                        data = scipy.io.loadmat(cons_folder1 + start_name + tech_name[testset]+'_1.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx][:,::4]

                                    elif specific_idx > 34.5:
                                        data = scipy.io.loadmat(cons_folder1 + start_name + tech_name[testset]+'_2.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx-35][:,::4]

                                
                                elif testset < 11.5:
                                    cons_folder_maj = "/scratch/users/magnusrk/Clean_flow/data/multiscore_agree_feb5/min_1/"
                                    start_name = 'eileen_feb5_cons1_'

                                    if specific_idx < 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_1.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx][:,::4]

                                    elif specific_idx > 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_2.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx-35][:,::4]

                                elif testset < 17.5:
                                    cons_folder_maj = "/scratch/users/magnusrk/Clean_flow/data/multiscore_agree_feb5/min_2/"
                                    start_name = 'eileen_feb5_cons2_'

                                    if specific_idx < 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_1.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx][:,::4]

                                    elif specific_idx > 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_2.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx-35][:,::4]

                                elif testset < 23.5:
                                    cons_folder_maj = "/scratch/users/magnusrk/Clean_flow/data/multiscore_agree_feb5/min_3/"
                                    start_name = 'eileen_feb5_cons3_'

                                    if specific_idx < 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_1.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx][:,::4]

                                    elif specific_idx > 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_2.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx-35][:,::4]


                                elif testset < 29.5:
                                    cons_folder_maj = "/scratch/users/magnusrk/Clean_flow/data/multiscore_agree_feb5/min_4/"
                                    start_name = 'eileen_feb5_cons4_'

                                    if specific_idx < 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_1.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx][:,::4]

                                    elif specific_idx > 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_2.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx-35][:,::4]


                                elif testset < 35.5:
                                    cons_folder_maj = "/scratch/users/magnusrk/Clean_flow/data/multiscore_agree_feb5/min_5/"
                                    start_name = 'eileen_feb5_cons5_'

                                    if specific_idx < 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_1.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx][:,::4]

                                    elif specific_idx > 34.5:
                                        data = scipy.io.loadmat(cons_folder_maj + start_name + tech_name[testset]+'_2.mat')
                                        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx-35][:,::4]




                                #else:
                                #    if specific_idx < 34.5:
                                #        data = scipy.io.loadmat(cons_folder_all + 'em_nov14_ini2_weigths1_' + tech_name[testset]+'_1.mat')
                                #        y_true = data['y_trues'][:,0:length[specific_idx],specific_idx]

                               #     elif specific_idx > 34.5:
                               #         data = scipy.io.loadmat(cons_folder_all + 'em_nov14_ini2_weigths1_' + tech_name[testset]+'_2.mat')
                               #         y_true = data['y_trues'][:,0:length[specific_idx],specific_idx-35]





                        s0 = multi_evaluation(
                            y_true.squeeze(), y_pred.squeeze(),
                            sfreq=pred_gen.fs, iou_ths=[0.0001])
                        
                        #np.savetxt("predictions/RES50_LSTM2_record_number"+str(r)+"op_AHI"+str(op_AHI)+"testset"+str(testset)+"_ypred.txt", y_pred,fmt='%.0f',delimiter = ",")
                        #np.savetxt("predictions/RES50_LSTM2_record_number"+str(r)+"op_AHI"+str(op_AHI)+"testset"+str(testset)+"_ytrue.txt", y_true,fmt='%.0f',delimiter = ",")
                        
                        #np.savetxt("probabilities/RES50_LSTM2_record_number"+str(r)+"op_AHI"+str(op_AHI)+"testset"+str(testset)+"_yprob.txt", y_prob,fmt='%1.5f',delimiter = ",")
                        
                        

                        scores_test0[r] = s0["f1_0"].values
                        scores_test1[r] = s0["precision_0"].values
                        scores_test2[r] = s0["recall_0"].values
                        scores_test3[r] = s0["n_match_0"].values
                        scores_test4[r] = s0["n_pred_0"].values
                        scores_test5[r] = s0["n_true_0"].values
                        

                        scores_test6[r] = s0["f1_1"].values
                        scores_test7[r] = s0["precision_1"].values
                        scores_test8[r] = s0["recall_1"].values
                        scores_test9[r] = s0["n_match_1"].values
                        scores_test10[r] = s0["n_pred_1"].values
                        scores_test11[r] = s0["n_true_1"].values
                        

                        scores_test12[r] = s0["f1_2"].values
                        scores_test13[r] = s0["precision_2"].values
                        scores_test14[r] = s0["recall_2"].values
                        scores_test15[r] = s0["n_match_2"].values
                        scores_test16[r] = s0["n_pred_2"].values
                        scores_test17[r] = s0["n_true_2"].values
                        
                        scores_test312[r] = s0["f1_3"].values
                        scores_test313[r] = s0["precision_3"].values
                        scores_test314[r] = s0["recall_3"].values
                        scores_test315[r] = s0["n_match_3"].values
                        scores_test316[r] = s0["n_pred_3"].values
                        scores_test317[r] = s0["n_true_3"].values


                        scores_testsamlet[r] = s0['n_match_all'].values
                        scores_predsamlet[r] = s0['n_pred_all'].values
                        scores_truesamlet[r] = s0['n_true_all'].values
                        scores_test0samlet[r] = s0['n_match_0a'].values
                        scores_test1samlet[r] = s0['n_match_1a'].values
                        scores_test2samlet[r] = s0['n_match_2a'].values
                        scores_test3samlet[r] = s0['n_match_3a'].values
                        
                        scores_test18[r] = s0["n_match_01"].values
                        scores_test19[r] = s0["n_match_02"].values
                        scores_test319[r] = s0["n_match_03"].values
                        scores_test20[r] = s0["n_match_10"].values
                        scores_test21[r] = s0["n_match_12"].values
                        scores_test321[r] = s0["n_match_13"].values
                        scores_test22[r] = s0["n_match_20"].values
                        scores_test23[r] = s0["n_match_21"].values
                        scores_test323[r] = s0["n_match_23"].values
                        scores_test3322[r] = s0["n_match_30"].values
                        scores_test3323[r] = s0["n_match_31"].values
                        scores_test3324[r] = s0["n_match_32"].values

                        scores_test51[r] = s0["avg_iou_0"].values
                        scores_test511[r] = s0["std_iou_0"].values
                        scores_test111[r] = s0["avg_iou_1"].values
                        scores_test1111[r] = s0["std_iou_1"].values
                        scores_test171[r] = s0["avg_iou_2"].values
                        scores_test1711[r] = s0["std_iou_2"].values
                        scores_test3171[r] = s0["avg_iou_3"].values
                        scores_test31711[r] = s0["std_iou_3"].values


                    torch.save(model.state_dict(), '/scratch/users/magnusrk/ABED/new_weights_and_model/saved_weights.pth')
                    torch.save(model, '/scratch/users/magnusrk/ABED/new_weights_and_model/saved_model.pth')

                    andre = pd.DataFrame()
                    #andre["record"] = [k for k in scores_test01.keys()]
                    andre["precision_0"] = [v for v in scores_test1.values()]
                    andre["recall_0"] = [v for v in scores_test2.values()]
                    andre["n_match_0"] = [v for v in scores_test3.values()]
                    andre["n_pred_0"] = [v for v in scores_test4.values()]
                    andre["n_true_0"] = [v for v in scores_test5.values()]
                    
                    andre["f1_1"] = [v for v in scores_test6.values()]
                    andre["precision_1"] = [v for v in scores_test7.values()]
                    andre["recall_1"] = [v for v in scores_test8.values()]
                    andre["n_match_1"] = [v for v in scores_test9.values()]
                    andre["n_pred_1"]  = [v for v in scores_test10.values()]
                    andre["n_true_1"] = [v for v in scores_test11.values()]
                    
                    andre["f1_2"] = [v for v in scores_test12.values()]
                    andre["precision_2"] = [v for v in scores_test13.values()]
                    andre["recall_2"] = [v for v in scores_test14.values()]
                    andre["n_match_2"] = [v for v in scores_test15.values()]
                    andre["n_pred_2"] = [v for v in scores_test16.values()]
                    andre["n_true_2"] = [v for v in scores_test17.values()]
                    
                    andre["f1_3"] = [v for v in scores_test312.values()]
                    andre["precision_3"] = [v for v in scores_test313.values()]
                    andre["recall_3"] = [v for v in scores_test314.values()]
                    andre["n_match_3"] = [v for v in scores_test315.values()]
                    andre["n_pred_3"] = [v for v in scores_test316.values()]
                    andre["n_true_3"] = [v for v in scores_test317.values()]

                    andre['n_match_all'] = [v for v in scores_testsamlet.values()]
                    andre['n_pred_all'] = [v for v in scores_predsamlet.values()]
                    andre['n_true_all'] = [v for v in scores_truesamlet.values()]
                    andre['n_match_0a'] = [v for v in scores_test0samlet.values()]
                    andre['n_match_1a'] = [v for v in scores_test1samlet.values()]
                    andre['n_match_2a'] = [v for v in scores_test2samlet.values()]
                    andre['n_match_3a'] = [v for v in scores_test3samlet.values()]
                    
                    andre["n_match_01"] = [v for v in scores_test18.values()]
                    andre["n_match_02"] = [v for v in scores_test19.values()]
                    andre["n_match_03"] = [v for v in scores_test319.values()]
                    andre["n_match_10"] = [v for v in scores_test20.values()]
                    andre["n_match_12"] = [v for v in scores_test21.values()]
                    andre["n_match_13"] = [v for v in scores_test321.values()]
                    andre["n_match_20"] = [v for v in scores_test22.values()]
                    andre["n_match_21"] = [v for v in scores_test23.values()]
                    andre["n_match_23"] = [v for v in scores_test323.values()]
                    andre["n_match_30"] = [v for v in scores_test3322.values()]
                    andre["n_match_31"] = [v for v in scores_test3323.values()]
                    andre["n_match_32"] = [v for v in scores_test3324.values()]

                    andre["avg_iou_0"] = [v for v in scores_test51.values()]
                    andre["std_iou_0"] = [v for v in scores_test511.values()]
                    andre["avg_iou_1"] = [v for v in scores_test111.values()]
                    andre["std_iou_1"] = [v for v in scores_test1111.values()]
                    andre["avg_iou_2"] = [v for v in scores_test171.values()]
                    andre["std_iou_2"] = [v for v in scores_test1711.values()]
                    andre["avg_iou_3"] = [v for v in scores_test3171.values()]
                    andre["std_iou_3"] = [v for v in scores_test31711.values()]






                    andre["precision_0"] = andre["precision_0"].str.get(0)
                    andre["recall_0"] = andre["recall_0"].str.get(0)
                    andre["n_match_0"] = andre["n_match_0"].str.get(0)
                    andre["n_pred_0"] = andre["n_pred_0"].str.get(0)
                    andre["n_true_0"] = andre["n_true_0"].str.get(0)
                    


                    andre["f1_1"] = andre["f1_1"].str.get(0)
                    andre["precision_1"] = andre["precision_1"].str.get(0)
                    andre["recall_1"] = andre["recall_1"].str.get(0)
                    andre["n_match_1"] = andre["n_match_1"].str.get(0)
                    andre["n_pred_1"] = andre["n_pred_1"].str.get(0)
                    andre["n_true_1"] = andre["n_true_1"].str.get(0)
                    

                    andre["f1_2"] = andre["f1_2"].str.get(0)
                    andre["precision_2"] = andre["precision_2"].str.get(0)
                    andre["recall_2"] = andre["recall_2"].str.get(0)
                    andre["n_match_2"] = andre["n_match_2"].str.get(0)
                    andre["n_pred_2"] = andre["n_pred_2"].str.get(0)
                    andre["n_true_2"] = andre["n_true_2"].str.get(0)
                    
                    andre["f1_3"] = andre["f1_3"].str.get(0)
                    andre["precision_3"] = andre["precision_3"].str.get(0)
                    andre["recall_3"] = andre["recall_3"].str.get(0)
                    andre["n_match_3"] = andre["n_match_3"].str.get(0)
                    andre["n_pred_3"] = andre["n_pred_3"].str.get(0)
                    andre["n_true_3"] = andre["n_true_3"].str.get(0)

                    andre['n_match_all'] = andre['n_match_all'].str.get(0)
                    andre['n_pred_all'] = andre['n_pred_all'].str.get(0)
                    andre['n_true_all'] = andre['n_true_all'].str.get(0)

                    andre['n_match_0a'] = andre['n_match_0a'].str.get(0)
                    andre['n_match_1a'] = andre['n_match_1a'].str.get(0)
                    andre['n_match_2a'] = andre['n_match_2a'].str.get(0)
                    andre['n_match_3a'] = andre['n_match_3a'].str.get(0)
                    

                    andre["n_match_01"] = andre["n_match_01"].str.get(0)
                    andre["n_match_02"] = andre["n_match_02"].str.get(0)
                    andre["n_match_03"] = andre["n_match_03"].str.get(0)
                    andre["n_match_10"] = andre["n_match_10"].str.get(0)
                    andre["n_match_12"] = andre["n_match_12"].str.get(0)
                    andre["n_match_13"] = andre["n_match_13"].str.get(0)
                    andre["n_match_20"] = andre["n_match_20"].str.get(0)
                    andre["n_match_21"] = andre["n_match_21"].str.get(0)
                    andre["n_match_23"] = andre["n_match_23"].str.get(0)
                    andre["n_match_30"] = andre["n_match_30"].str.get(0)
                    andre["n_match_31"] = andre["n_match_31"].str.get(0)
                    andre["n_match_32"] = andre["n_match_32"].str.get(0)

                    andre["avg_iou_0"] = andre["avg_iou_0"].str.get(0)
                    andre["std_iou_0"] = andre["std_iou_0"].str.get(0)
                    andre["avg_iou_1"] = andre["avg_iou_1"].str.get(0)
                    andre["std_iou_1"] = andre["std_iou_1"].str.get(0)
                    andre["avg_iou_2"] = andre["avg_iou_2"].str.get(0)
                    andre["std_iou_2"] = andre["std_iou_2"].str.get(0)
                    andre["avg_iou_3"] = andre["avg_iou_3"].str.get(0)
                    andre["std_iou_3"] = andre["std_iou_3"].str.get(0)


                    #save scores for each setup
                    scores_test0 = pd.DataFrame(scores_test0)
                    scores_test0 = scores_test0.fillna(0.)
                    scores_test0 = scores_test0.transpose()
                    scores_test0.columns = ["f1_0"]
                    scores_test0["record"] = scores_test0.index
                    scores_test0 = scores_test0.reset_index(drop=True)
                    scores_test0["duration"] = duration


                    score999 = pd.concat([scores_test0, andre], axis=1)#, join_axes=[scores_test0.record])


                    score999.to_csv('results/'+start_name2+'leftout'+tech_name[testset]+'_'+start_name+"{}dur_{}nchn_{}lr_{}epochs_{}drop_testset{}_split{}_opahi{}_lag{}_window{}_linear{}.csv".format(
                         duration,n_channels,lr,epochs,dropout,testset,idx_split,op_AHI,lagg,window,linearlayer),sep=';')

                    score999_eval =pd.DataFrame({'F1':[score999.f1_0[score999.n_true_0!=0].mean(0), score999.f1_1[score999.n_true_1!=0].mean(0), score999.f1_2[score999.n_true_2!=0].mean(0)],
                                                 'Precision':[score999.precision_0[score999.n_true_0!=0].mean(0), score999.precision_1[score999.n_true_1!=0].mean(0), score999.precision_2[score999.n_true_2!=0].mean(0)],
                                                 'Recall':[score999.recall_0[score999.n_true_0!=0].mean(0), score999.recall_1[score999.n_true_1!=0].mean(0), score999.recall_2[score999.n_true_2!=0].mean(0)]})
                    score999_eval.to_csv('results/Magnus_test',sep=';')


                # prediction
                ss=2

                if 1==ss:

                    print("\nstaring prediction on testing data")
                    scores_test0 = []

                    for idx_r, r in enumerate(tqdm(r_test)):
                        print("\nrecord {}".format(r))

                        test_gen = DataGen(
                            df[df.record == r],
                            index_on_events=False, ratio_positive=None,
                            **gen_params)

                        y_true, y_pred, y_prob = model.predict_generator(
                            test_gen, detection_threshold=[thres0,thres1,thres2])

                        y_true_ = y_true

                        # if df[df.record == r].label.values[0] == 0:
                         #elif df[df.record == r].label.values[0] == 1:
                       # if 1 == 1:# df[df.record == r].label.values[0] == 2
                        y_true_ = y_true_[:, ::downsampling]

                        #y_pred0 = y_pred[:, 0, :]
                        #y_pred1 = y_pred[:, 1, :]
                        #y_pred2 = y_pred[:, 2, :]

                        #y_pred0 = y_pred0[:, ::downsampling]
                        #y_pred1 = y_pred1[:, ::downsampling]
                        #y_pred2 = y_pred2[:, ::downsampling]

                        y_pred[0,1,:]=y_pred[1,1,:]
                        y_pred[0,2,:]=y_pred[2,2,:]
                        y_pred[0,3,:]=y_pred[3,3,:]
                        
                        
                        y_pred[0,0,:] = y_pred[0,0,:] - np.multiply(y_pred[0,0,:],y_pred[0,1,:])
                        y_pred[0,3,:] = y_pred[0,3,:] - np.multiply(y_pred[0,3,:],y_pred[0,1,:])- np.multiply(y_pred[0,3,:],y_pred[0,0,:])
                        y_pred[0,2,:]= y_pred[0,2,:] - np.multiply(y_pred[0,2,:], y_pred[0,1,:])- np.multiply(y_pred[0,2,:], y_pred[0,0,:]) - np.multiply(y_pred[0,2,:],y_pred[0,3,:])
                        
                        
                        y_pred = y_pred[0,:,:]

                        y_pred0 = y_pred[0,:]
                        y_pred1 = y_pred[1,:]
                        y_pred2 = y_pred[2,:]

                        if 100000 > r > 10000: #Eillen's multi scored
                                specific_idx = r-10001
                                length = [1255680, 1289280, 1230720, 1059840, 1085760, 1247040, 1028160, 1292160, 1038720, 1093440, 1224000, 965760, 1314240, 1264320, 1101120, 1070400, 1180800, 1351680, 1005120 ,1122240 ,1138560, 1083840, 1098240, 998400, 1257600, 1080960, 1032000, 1176000, 1140480, 1127040, 1097280, 1235520, 1181760, 1082880, 964800, 1185600, 1104960, 1273920, 1171200, 1061760, 1119360, 1128960, 1268160, 1297920, 948480, 971520, 1130880, 1351680, 1142400, 1222080, 1265280, 1015680, 1134720, 1262400, 1085760, 1245120, 1209600, 1198080, 1048320, 1194240, 1264320, 1704960, 1233600, 1229760, 1060800, 1105920, 1194240, 1113600, 988800, 1123200]


                                if specific_idx < 34.5:
                                    data = scipy.io.loadmat(cons_folder_all + 'em_nov14_ini2_weigths1_' + tech_name[7]+'_1.mat')
                                    y_true = data['y_trues'][:,0:length[specific_idx],specific_idx][:,::4]

                                elif specific_idx > 34.5:
                                    data = scipy.io.loadmat(cons_folder_all + 'em_nov14_ini2_weigths1_' + tech_name[7]+'_2.mat')
                                    y_true = data['y_trues'][:,0:length[specific_idx],specific_idx-35][:,::4]

                        prediction0 = dict(y_true0=y_true0, y_true1=y_true1, y_true2=y_true2, 
                                        y_pred0=y_pred0, y_pred1=y_pred1, y_pred2=y_pred2, 
                                        data=test_gen.eegs[r]["data"])

                        pred_path = "/scratch/users/jthybo/mat_wsc_4chn_dec3/"

                        savemat("{}_{}.mat".format(pred_path,
                                                    str(r)), mdict=prediction0)

                        del y_true_

                del scores_test0#, model #was scores_val0
                
                
