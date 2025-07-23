import argparse
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import json

from datagenerator.datagen import DataGen
from model.event_detector import EventDetector
from metrics.metrics import multi_evaluation
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Train Event Detector model")
    parser.add_argument("--data_input", type=str, required=True, help="Path to input .csv file")
    parser.add_argument("--model_out_path", type=str, required=True, help="Path to save the model .pth")
    parser.add_argument("--thresholds_out_path", type=str, required=True, help="Path to save the thresholds .json")
    parser.add_argument("--histories_out_path", type=str, required=True, help="Path to save training history")
    parser.add_argument("--weights_out_dir", type=str, required=True, help="Directory to save weights")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    df = pd.read_csv(args.data_input)
    df = df[df.record.notna()]  # drop rows with NaN record
    df = df[df.record != 3347]  # add other exclusions as needed

    sfreq = pd.unique(df["fs"])[0]
    window = 240
    selected_channels = ["Flow", "Therm", "Thor", "Abdo", "SpO2", "Arousal", "Wake"]
    channels = selected_channels
    downsampling = 1
    minimum_overlap = 0.5
    detection_thresholds = [0.91, 0.81, 0.71, 0.61, 0.51]
    idx_split = 30

    records = sorted(pd.unique(df.record))
    r_train, r_val = train_test_split(records, train_size=0.9, test_size=0.1, random_state=idx_split)

    df_train = df[df.record.isin(r_train)].reset_index(drop=True)
    df_val = df[df.record.isin(r_val)].reset_index(drop=True)

    # Label balancing factors
    n_HYPO = df[df['label'] == 2]['n_events'].sum()
    factor_HYPO = 1
    factor_NAD_HYPO = n_HYPO / df[df['label'] == 3]['n_events'].sum()
    factor_OSA = n_HYPO / df[df['label'] == 0]['n_events'].sum()
    factor_CSA = n_HYPO / df[df['label'] == 1]['n_events'].sum()

    gen_params = dict(
        selected_channels=selected_channels,
        window=window,
        downsampling=downsampling,
        minimum_overlap=minimum_overlap,
        channels=channels
    )

    train_gen = DataGen(df_train, **gen_params,
                        index_on_events=True,
                        ratio_positive=factor_OSA,
                        ratio_positive1=factor_CSA,
                        ratio_positive2=factor_HYPO,
                        ratio_positive3=factor_NAD_HYPO,
                        traindata_factor=1)

    val_gen = DataGen(df_val, **gen_params,
                      index_on_events=True,
                      ratio_positive=factor_OSA,
                      ratio_positive1=factor_CSA,
                      ratio_positive2=factor_HYPO,
                      ratio_positive3=factor_NAD_HYPO,
                      traindata_factor=1)

    model = EventDetector(
        n_channels=len(selected_channels),
        n_classes=4,
        n_freq_channels=0,
        freq_factor=[1]*len(selected_channels),
        n_times=window * sfreq,
        num_workers=10,
        fs=sfreq,
        histories_path=args.histories_out_path,
        weights_path=os.path.join(args.weights_out_dir, "final_weights.pth"),
        loss="worst_negative_mining",
        default_event_sizes=[i * window * sfreq for i in range(1, 13)],
        factor_overlap=2,
        lr=0.001,
        patience=100,
        lr_decrease_patience=30,
        epochs=8,
        k_max=5,
        max_pooling=2,
        batch_size=128,
        dropout=0.9,
        partial_eval=-1,
        linearlayer=0,
        RES_architecture=[3, 4, 6, 3]
    )

    model.fit_generator(train_gen, val_gen)

    thresholds = {}
    for label in range(4):
        scores_label = {}
        for r in tqdm(r_val):
            if df[df.record == r].n_events.values[0] == 0:
                continue

            pred_gen = DataGen(df[df.record == r], **gen_params, index_on_events=False)
            y_true, y_pred, _ = model.predict_generator(pred_gen, detection_thresholds)

            for idx, th in enumerate(detection_thresholds):
                s = multi_evaluation(y_true.squeeze(), y_pred[idx].squeeze(), sfreq=pred_gen.fs, iou_ths=[0.001])
                metric_key = f"f1_{label}"
                scores_label.setdefault(th, []).append(s[metric_key].values[0])

        avg_scores = {th: np.mean(vals) for th, vals in scores_label.items()}
        best_th = max(avg_scores, key=avg_scores.get)
        thresholds[f"class_{label}"] = best_th

    # Save model
    torch.save(model.state_dict(), args.model_out_path)

    # Save thresholds
    with open(args.thresholds_out_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print("Training complete. Thresholds saved to:", args.thresholds_out_path)

if __name__ == "__main__":
    main()