import pandas as pd
import numpy as np


def hp_selection(df_val, metric='f1', hp=["clf_th"]):

    # selecting hp
    hp_val = df_val.groupby(hp).mean().reset_index()
    metric_max = hp_val[metric].max()
    hp = hp_val[hp_val[metric] == metric_max][hp]

    return hp


def get_iou_matrix(df_true, df_pred):
    """get_iou_matrix
    """

    start_true = df_true.start.values.reshape(-1, 1)
    start_pred = df_pred.start.values.reshape(1, -1)

    end_true = df_true.end.values.reshape(-1, 1)
    end_pred = df_pred.end.values.reshape(1, -1)

    x_a = np.maximum(start_true, start_pred)
    x_b = np.minimum(end_true, end_pred)

    inter_area = (x_b - x_a)

    area_a = end_true - start_true
    area_b = end_pred - start_pred

    iou = inter_area / (area_a + area_b - inter_area)

    return iou


def compute_per_stage_metrics(df_true, df_pred, stages, sfreq):
    """compute_per_stage_metrics
    """

    df = []
    for label in [0, 1, 2, 3, 4]:

        stages_ = stages.copy()
        stages_[stages_ != label] = 0
        stages_[stages_ == label] = 1

        delta = stages_[1:] - stages_[:-1]

        idx_start, = np.where(delta == 1)
        idx_end, = np.where(delta == -1)

        df_stages_ = pd.DataFrame()
        df_stages_["idx_start"] = idx_start
        df_stages_["idx_end"] = idx_end
        df_stages_["start"] = df_stages_["idx_start"] / sfreq
        df_stages_["end"] = df_stages_["idx_end"] / sfreq
        df_stages_["duration"] = df_stages_["end"] - df_stages_["start"]

        iou_ = get_iou_matrix(df_true, df_stages_)
        count_true = (iou_ > 0).astype(int)
        a = np.sum(count_true, axis=1)
        a[a > 0] = 1
        n_match_true = np.sum(a)

        iou_ = get_iou_matrix(df_pred, df_stages_)
        count_pred = (iou_ > 0).astype(int)
        a = np.sum(count_pred, axis=1)
        a[a > 0] = 1
        n_match_pred = np.sum(a)

        if n_match_true != 0:
            iou = get_iou_matrix(
                df_true.iloc[count_true.sum(axis=1) != 0],
                df_pred.iloc[count_pred.sum(axis=1) != 0])

            count = (iou > 0.3).astype(int)

            n_match = np.sum(count)

            n_pos = iou.shape[1]
            n_rel = iou.shape[0]

            precision = n_match / n_pos
            recall = n_match / n_rel

            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            n_match = -1
            n_pos = -1
            n_rel = -1

            precision = -1
            recall = -1
            f1 = -1

        df_ = dict()

        df_["precision"] = [precision]
        df_["recall"] = [recall]
        df_["f1"] = [f1]

        df_["n_pos"] = [n_pos]
        df_["n_rel"] = [n_rel]
        df_["n_match"] = [n_match]
        df_["stage"] = [label]
        df_["n_pred"] = n_match_pred
        df_["n_true"] = n_match_true
        df_["IoU"] = 0.3
        df_["by_sample_f1"] = -1
        df_["by_sample_precision"] = -1
        df_["by_sample_recall"] = -1

        df.append(pd.DataFrame(df_))

    return pd.concat(df)


def evaluation(
        true, pred, sfreq=256,
        iou_ths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        stages=None):
    """
    """

    # predictions
    a = np.zeros(pred.shape[0] + 1)
    b = np.zeros(pred.shape[0] + 1)
    a[1:] = pred
    b[:-1] = pred
    delta = b - a
    # delta = pred[1:] - pred[:-1]

    idx_start, = np.where(delta == 1)
    idx_end, = np.where(delta == -1)

    sss=2
    if sss==2:

        # Remove events 
         # removing long events 
        long_event = np.where(idx_end-idx_start>60*2.5*sfreq)
        idx_start=np.delete(idx_start, long_event)
        idx_end=np.delete(idx_end, long_event)

        if len(idx_start)>1.5:

            #Merging events
            merge_event = np.where(idx_start[1:]-idx_end[:-1]<2*sfreq)
          
            idx_end = np.delete(idx_end, merge_event)
            new = np.zeros(len(idx_end))
            new[0]=idx_start[0]
            new[1:] = np.delete(idx_start[1:], merge_event)
            idx_start = new  
            #idx_start[1:] =  np.delete(idx_start[1:], merge_event)
            #idx_start=np.delete(idx_start, merge_event.values+1)
            #idx_end=np.delete(idx_end, merge_event.values)


        # Remove short events 
         # removing short duration
        short_event = np.where(idx_end-idx_start<8*sfreq)
        idx_start=np.delete(idx_start, short_event)
        idx_end=np.delete(idx_end, short_event)




    df_pred = pd.DataFrame()
    df_pred["idx_start"] = idx_start
    df_pred["idx_end"] = idx_end
    df_pred["start"] = df_pred["idx_start"] / sfreq
    df_pred["end"] = df_pred["idx_end"] / sfreq
    df_pred["duration"] = df_pred["end"] - df_pred["start"]
    df_pred["scorer"] = "model"

    # predictions
    a = np.zeros(pred.shape[0] + 1)
    b = np.zeros(pred.shape[0] + 1)
    a[1:] = true
    b[:-1] = true
    delta = b - a

    idx_start, = np.where(delta == 1)
    idx_end, = np.where(delta == -1)

    df_true = pd.DataFrame()
    df_true["idx_start"] = idx_start
    df_true["idx_end"] = idx_end
    df_true["start"] = df_true["idx_start"] / sfreq
    df_true["end"] = df_true["idx_end"] / sfreq
    df_true["duration"] = df_true["end"] - df_true["start"]
    df_true["scorer"] = "human"

    df = pd.concat([df_true, df_pred])

    df_true = df[df.scorer == "human"]
    df_pred = df[df.scorer == "model"]

    # evaluate pairwise IoU
    iou = get_iou_matrix(df_true, df_pred)

    precisions = np.zeros(len(iou_ths))
    recalls = np.zeros(len(iou_ths))
    n_positives = np.zeros(len(iou_ths))
    n_relevants = np.zeros(len(iou_ths))
    n_matches = np.zeros(len(iou_ths))
    #start_stop_discrepancy = {iou_thr: None for iou_thr in iou_ths}
    avg_iou = np.zeros(len(iou_ths))
    std_iou = np.zeros(len(iou_ths))

    #start_err_avg = np.zeros(len(iou_ths))
    #start_err_std = np.zeros(len(iou_ths))
    #end_err_avg = np.zeros(len(iou_ths))
    #end_err_std = np.zeros(len(iou_ths)) 


    for i, iou_th in enumerate(iou_ths):
        iou_match = iou[iou>=iou_th]
        count = (iou >= iou_th).astype(int)
        keep_idx = [True] * count.shape[1]
        if len(iou_match.shape)>1.5:
            iou_match = iou_match[:,0]
        #if isinstance(iou_match[1],list):
         #   iou_match = iou_match[:,1]=[]
       # for idx in np.where(np.sum(count, axis=1) > 1)[0]:
        #    max_iou = np.max(iou[idx, :])
         #   for j in np.where(count[idx, :])[0]:
          #      if iou[idx, j] < max_iou:
           #         keep_idx[j] = False


        count1 = iou_match
        n_fp = len(np.sum(count, axis=0)[np.sum(count, axis=0)==0])
        n_fn = len(np.sum(count, axis=1)[np.sum(count, axis=1)==0])
        # to avoid counting twice an event

        #match_pred = np.sum(count, axis=1)
        #match_true = np.sum(count, axis=0)
        count = np.sum(count, axis=1)

        #match_true[match_true > 0] = 1
        #match_pred[match_pred > 0] = 1
        count[count > 0] = 1


        # Find current start/stop discrepancies and save them as [num_events, 2] arrays
        #try:
         #   start_discrepancy = (df_pred.loc[(np.sum((iou >= iou_th), axis=0) == 1) & keep_idx, 'start'].values - df_true.loc[count, 'start'].values)[:, np.newaxis]
          #  end_discrepancy = (df_pred.loc[(np.sum((iou >= iou_th), axis=0) == 1) & keep_idx, 'end'].values - df_true.loc[count, 'end'].values)[:, np.newaxis]
           # start_stop_discrepancy[iou_th] = np.concatenate([start_discrepancy, end_discrepancy], axis=1)
            #start_err_avg[i] = np.mean(start_discrepancy)
       #     start_err_std[i] = np.std(start_discrepancy)
        #    end_err_avg[i] = np.mean(end_discrepancy)
         #   end_err_std[i] = np.std(end_discrepancy)
        #except TypeError:
         #   pass




        n_match = np.sum(count)

        n_pos = iou.shape[1]
        n_rel = iou.shape[0]

        avg_iou[i] = np.sum(iou_match)/n_match
        std_iou[i] = np.std(iou_match)

        n_positives[i] = n_pos
        n_relevants[i] = n_rel
        n_matches[i] = n_match
        precisions[i] = n_match / (n_match + n_fp)
        recalls[i] = n_match / (n_match + n_fn)

    # compute f1 score
    f1 = 2 * (recalls * precisions) / (recalls + precisions)

    df = pd.DataFrame()
    df["precision"] = precisions
    df["recall"] = recalls
    df["f1"] = f1
    df["IoU"] = iou_ths
    df["n_pos"] = n_positives
    df["n_rel"] = n_relevants
    df["n_match"] = n_matches
    df["n_pred"] = n_positives
    df["n_true"] = n_relevants
    df["avg_iou"] = avg_iou
    df["std_iou"] = std_iou

   # # Add start/stop errors
    #df['start_err_avg'] = start_err_avg
    #df['start_err_std'] = start_err_std
    #df['end_err_avg'] = end_err_avg
    #df['end_err_std'] = end_err_std 

    # by sample metric
    y_match, = np.where(true + pred == 2)
    n_match = y_match.shape[0]

    n_pos = np.sum(pred)
    n_rel = np.sum(true)

    df["by_sample_precision"] = n_match / n_pos
    df["by_sample_recall"] = n_match / n_rel

    df["by_sample_f1"] = 2 * (
        df["by_sample_precision"] * df["by_sample_recall"]) / \
        (df["by_sample_precision"] + df["by_sample_recall"])
    df["stage"] = "NO"

    if stages is not None:
        df_ = compute_per_stage_metrics(df_true, df_pred, stages, sfreq)

        df = pd.concat([df, df_])

    return df#, start_stop_discrepancy 

def multi_evaluation(
        true, pred, sfreq=256,
        iou_ths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    y_true0 = true[0, :]
    y_pred0 = pred[ 0, :]

    # elif df[df.record == r].label.values[0] == 1:
    y_true1 = true[1, :]
    y_pred1 = pred[ 1, :]
    # elif df[df.record == r].label.values[0] == 2:
    y_true2 = true[2, :]
    y_pred2 = pred[ 2, :]
    
    y_true3 = true[3, :]
    y_pred3 = pred[ 3, :]

    y_true_samlet = y_true0 + y_true1 + y_true2 + y_true3
    y_pred_samlet = y_pred0 + y_pred1 + y_pred2 + y_pred3

    y_true_samlet[y_true_samlet>1]=1
    y_pred_samlet[y_pred_samlet>1]=1

    #for 0
    df00 = evaluation(
        y_true0.squeeze(), y_pred0.squeeze(),
        sfreq, iou_ths)
    df01 = evaluation(
        y_true0.squeeze(), y_pred1.squeeze(),
        sfreq, iou_ths)
    df02 = evaluation(
        y_true0.squeeze(), y_pred2.squeeze(),
        sfreq, iou_ths)
    df03 = evaluation(
        y_true0.squeeze(), y_pred3.squeeze(),
        sfreq, iou_ths)

    #for 1
    df10 = evaluation(
        y_true1.squeeze(), y_pred0.squeeze(),
        sfreq, iou_ths)
    df11 = evaluation(
        y_true1.squeeze(), y_pred1.squeeze(),
        sfreq, iou_ths)
    df12 = evaluation(
        y_true1.squeeze(), y_pred2.squeeze(),
        sfreq, iou_ths)
    df13 = evaluation(
        y_true1.squeeze(), y_pred3.squeeze(),
        sfreq, iou_ths)

    # for 2
    df20 = evaluation(
        y_true2.squeeze(), y_pred0.squeeze(),
        sfreq, iou_ths)
    df21 = evaluation(
        y_true2.squeeze(), y_pred1.squeeze(),
        sfreq, iou_ths)
    df22 = evaluation(
        y_true2.squeeze(), y_pred2.squeeze(),
        sfreq, iou_ths)
    df23 = evaluation(
        y_true2.squeeze(), y_pred3.squeeze(),
        sfreq, iou_ths)
    
    # for 3
    df30 = evaluation(
        y_true3.squeeze(), y_pred0.squeeze(),
        sfreq, iou_ths)
    df31 = evaluation(
        y_true3.squeeze(), y_pred1.squeeze(),
        sfreq, iou_ths)
    df32 = evaluation(
        y_true3.squeeze(), y_pred2.squeeze(),
        sfreq, iou_ths)
    df33 = evaluation(
        y_true3.squeeze(), y_pred3.squeeze(),
        sfreq, iou_ths)

    #for samlet
    df_samlet = evaluation(
        y_true_samlet.squeeze(), y_pred_samlet.squeeze(),
        sfreq, iou_ths)

    df_0samlet = evaluation(
        y_true_samlet.squeeze(), y_pred0.squeeze(),
        sfreq, iou_ths)
    df_1samlet = evaluation(
        y_true_samlet.squeeze(), y_pred1.squeeze(),
        sfreq, iou_ths)
    df_2samlet = evaluation(
        y_true_samlet.squeeze(), y_pred2.squeeze(),
        sfreq, iou_ths)
    df_3samlet = evaluation(
        y_true_samlet.squeeze(), y_pred3.squeeze(),
        sfreq, iou_ths)


    df = pd.DataFrame()
    df["op_ahi_0"] = abs(df00["precision"]-df00["recall"])
    df["precision_0"] = df00["precision"]
    df["recall_0"] = df00["recall"]
    df["f1_0"] = df00["f1"]
    df["n_match_0"] = df00["n_match"]
    df["n_pred_0"] = df00["n_pred"]
    df["n_true_0"] = df00["n_true"]
    df["avg_iou_0"] = df00["avg_iou"]
    df["std_iou_0"] = df00["std_iou"]
    #df['start_err_avg_0'] = df00["start_err_avg"]
    #df['start_err_std_0'] = df00["start_err_avg"]
    #df['end_err_avg_0'] = df00["start_err_avg"]
    #df['end_err_std_0'] = df00["start_err_avg"]

    df["op_ahi_1"] = abs(df11["precision"]-df11["recall"])
    df["precision_1"] = df11["precision"]
    df["recall_1"] = df11["recall"]
    df["f1_1"] = df11["f1"]
    df["n_match_1"] = df11["n_match"]
    df["n_pred_1"] = df11["n_pred"]
    df["n_true_1"] = df11["n_true"]
    df["avg_iou_1"] = df11["avg_iou"]
    df["std_iou_1"] = df11["std_iou"]
    #df['start_err_avg_1'] = df11["start_err_avg"]
    #df['start_err_std_1'] = df11["start_err_avg"]
    #df['end_err_avg_1'] = df11["start_err_avg"]
    #df['end_err_std_1'] = df11["start_err_avg"]
     
    df["op_ahi_2"] = abs(df22["precision"]-df22["recall"])
    df["precision_2"] = df22["precision"]
    df["recall_2"] = df22["recall"]
    df["f1_2"] = df22["f1"]
    df["n_match_2"] = df22["n_match"]
    df["n_pred_2"] = df22["n_pred"]
    df["n_true_2"] = df22["n_true"]
    df["avg_iou_2"] = df22["avg_iou"]
    df["std_iou_2"] = df22["std_iou"]
    #df['start_err_avg_2'] = df22["start_err_avg"]
    #df['start_err_std_2'] = df22["start_err_avg"]
    #df['end_err_avg_2'] = df22["start_err_avg"]
    #df['end_err_std_2'] = df22["start_err_avg"]
    
    df["op_ahi_3"] = abs(df33["precision"]-df22["recall"])
    df["precision_3"] = df33["precision"]
    df["recall_3"] = df33["recall"]
    df["f1_3"] = df33["f1"]
    df["n_match_3"] = df33["n_match"]
    df["n_pred_3"] = df33["n_pred"]
    df["n_true_3"] = df33["n_true"]
    df["avg_iou_3"] = df33["avg_iou"]
    df["std_iou_3"] = df33["std_iou"]
    #df['start_err_avg_2'] = df22["start_err_avg"]
    #df['start_err_std_2'] = df22["start_err_avg"]
    #df['end_err_avg_2'] = df22["start_err_avg"]
    #df['end_err_std_2'] = df22["start_err_avg"]

    df['n_match_all']= df_samlet['n_match']
    df['n_pred_all']= df_samlet['n_pred']
    df['n_true_all']= df_samlet['n_true']
    df['n_match_0a']= df_0samlet['n_match']
    df['n_match_1a']= df_1samlet['n_match']
    df['n_match_2a']= df_2samlet['n_match']
    df['n_match_3a']= df_3samlet['n_match']

    df["n_match_01"] = df01["n_match"]
    df["n_match_02"] = df02["n_match"]
    df["n_match_03"] = df03["n_match"]
    df["n_match_10"] = df10["n_match"]
    df["n_match_12"] = df12["n_match"]
    df["n_match_13"] = df13["n_match"]
    df["n_match_20"] = df20["n_match"]
    df["n_match_21"] = df21["n_match"]
    df["n_match_23"] = df23["n_match"]
    df["n_match_30"] = df30["n_match"]
    df["n_match_31"] = df31["n_match"]
    df["n_match_32"] = df32["n_match"]

    return df



