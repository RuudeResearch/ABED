#!/usr/bin/env python3
"""
Sleep Data Preprocessing Script

Command-line tool for preprocessing sleep EDF and annotation files into memory-mapped format

Usage:
    python sleep_preprocessing.py --input_csv /path/to/records.csv --arousal_path /path/to/arousals/ --target_path /path/to/output/
    
    python sleep_preprocessing.py --input_csv /path/to/records.csv --arousal_path /path/to/arousals/ --target_path /path/to/output/ --start_idx 0 --end_idx 100

@author: Magnus Ruud Kjær
Updated: 2025
"""

import os
import pandas as pd
import xmltodict
import numpy as np
import argparse
import sys
from pathlib import Path
from mne.io.edf import read_raw_edf
from tqdm import tqdm
import xml.etree.ElementTree as ET
from scipy.signal import butter, sosfiltfilt, resample
from scipy import signal
import copy


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sleep Data Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sleep_preprocessing.py --input_csv /data/records.csv --arousal_path /data/arousals/ --target_path /output/processed/
    
    python sleep_preprocessing.py --input_csv /data/records.csv --arousal_path /data/arousals/ --target_path /output/processed/ --start_idx 0 --end_idx 100 --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input_csv', 
        type=str, 
        required=True,
        help='Path to CSV file containing record information (with edf_f and annot_f columns)'
    )
    
    parser.add_argument(
        '--arousal_path', 
        type=str, 
        required=True,
        help='Base path to arousal files directory'
    )
    
    parser.add_argument(
        '--target_path', 
        type=str, 
        required=True,
        help='Target directory for processed output files'
    )
    
    # Optional processing parameters
    parser.add_argument(
        '--start_idx', 
        type=int, 
        default=0,
        help='Starting index for processing records (default: 0)'
    )
    
    parser.add_argument(
        '--end_idx', 
        type=int, 
        default=None,
        help='Ending index for processing records (default: process all)'
    )
    
    parser.add_argument(
        '--target_freq', 
        type=int, 
        default=8,
        help='Target sampling frequency (default: 8 Hz)'
    )
    
    # Event type name configurations
    parser.add_argument(
        '--central_apnea_names', 
        type=str, 
        nargs='+',
        default=['Central apnea|Central Apnea'],
        help='Names for central apnea events (default: "Central apnea|Central Apnea")'
    )
    
    parser.add_argument(
        '--obstructive_apnea_names', 
        type=str, 
        nargs='+',
        default=['Obstructive apnea|Obstructive Apnea'],
        help='Names for obstructive apnea events (default: "Obstructive apnea|Obstructive Apnea")'
    )
    
    parser.add_argument(
        '--hypopnea_names', 
        type=str, 
        nargs='+',
        default=['Hypopnea|Hypopnea', 'Unsure|Unsure'],
        help='Names for hypopnea events (default: "Hypopnea|Hypopnea" "Unsure|Unsure")'
    )
    
    parser.add_argument(
        '--arousal_names', 
        type=str, 
        nargs='+',
        default=['Arousal|Arousal ()', 'Spontaneous arousal|Arousal (ARO SPONT)', 
                'ASDA arousal|Arousal (ASDA)', 'Arousal resulting from respiratory effort|Arousal (ARO RES)'],
        help='Names for arousal events'
    )
    
    parser.add_argument(
        '--desaturation_names', 
        type=str, 
        nargs='+',
        default=['SpO2 desaturation|SpO2 desaturation'],
        help='Names for desaturation events (default: "SpO2 desaturation|SpO2 desaturation")'
    )
    
    parser.add_argument(
        '--desaturation_threshold', 
        type=float, 
        default=3.0,
        help='Desaturation threshold (default: 3.0)'
    )
    
    # Channel configuration
    parser.add_argument(
        '--channels', 
        type=str, 
        nargs='+',
        default=['Flow', 'Therm', 'Thor', 'Abdo', 'SpO2'],
        help='Signal channels to process (default: Flow Therm Thor Abdo SpO2)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    # Check input CSV exists
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
    
    # Check arousal path exists
    if not os.path.exists(args.arousal_path):
        raise FileNotFoundError(f"Arousal path not found: {args.arousal_path}")
    
    # Create target path if it doesn't exist
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)
        print(f"Created target directory: {args.target_path}")
    
    # Validate indices
    if args.start_idx < 0:
        raise ValueError("start_idx must be non-negative")
    
    if args.end_idx is not None and args.end_idx <= args.start_idx:
        raise ValueError("end_idx must be greater than start_idx")
    
    # Validate target frequency
    if args.target_freq <= 0:
        raise ValueError("target_freq must be positive")
    
    # Validate desaturation threshold
    if args.desaturation_threshold < 0:
        raise ValueError("desaturation_threshold must be non-negative")


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply butterworth bandpass filter"""
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def transform_SpO2(x):
    x_in = np.array([60, 100])
    y_out = np.array([-1, 1])
    m = (y_out[1] - y_out[0]) / (x_in[1] - x_in[0])
    b = y_out[0] - m * x_in[0]
    y = m * x + b
    return y

def interp_downsample(x,fs_old,fs_new):
    
    time_new = np.linspace(0,len(x[0,:]),int(fs_new/fs_old*len(x[4,:])))
    time_old = np.linspace(0,len(x[4,:]),len(x[4,:]))
    y = np.zeros((5,len(time_new)))
    y[0,:] = np.interp(time_new, time_old, x[0,:])
    y[1,:] = np.interp(time_new, time_old, x[1,:])
    y[2,:] = np.interp(time_new, time_old, x[2,:])
    y[3,:] = np.interp(time_new, time_old, x[3,:])
    y[4,:] = transform_SpO2(np.concatenate((np.zeros(int((fs_new/2)-1)), np.round(moving_average(np.interp(time_new, time_old, x[4,:]),fs_new)), np.zeros(int(fs_new/2)))))
    return y 


def load_arousal_data(arousal_path, annot_file, target_freq, verbose=False):
    """Load arousal and wake data"""
    arousal_file = os.path.join(arousal_path, annot_file.split('/')[-1][:-9] + '.txt')
    
    if verbose:
        print(f"Loading arousal data from: {arousal_file}")
    
    try:
        arousal_data = np.loadtxt(arousal_file, dtype=None, delimiter=',')
        arousal = np.repeat(arousal_data[0, :], target_freq)
        wake = np.repeat(arousal_data[1, :], target_freq)
        
        if np.isnan(arousal).any() or np.isnan(wake).any():
            raise ValueError("NaN values found in arousal data")
        
        return arousal, wake
    
    except OSError:
        raise FileNotFoundError(f"Arousal file not found: {arousal_file}")


def process_signal_data(raw, channels, target_freq, verbose=False):
    """Process and filter signal data"""
    if verbose:
        print("Processing signal data...")
    
    # Extract channel data
    try:
        x = raw.get_data(picks=[raw.ch_names.index(chn) for chn in channels])
    except ValueError as e:
        raise ValueError(f"Error extracting channels: {e}")
    
    original_fs = raw._raw_lengths[0] / np.ceil(raw._last_time)
    
    # Apply bandpass filtering to first 4 channels (Flow, Therm, Thor, Abdo)
    for i in range(4):
        x[i, :] = butter_bandpass_filter(x[i, :], 0.025, 4, original_fs, order=4)
    
    # Normalize first 4 channels
    mean4 = np.mean(x[0:4, :], axis=1)
    std4 = np.std(x[0:4, :], axis=1)
    
    for i in range(4):
        x[i, :] = (x[i, :] - mean4[i]) / std4[i]
    
    # Process SpO2 channel (index 4)
    if x.shape[0] > 4:
        new = copy.deepcopy(x[4, :])
        new[new < 60] = np.nan
        x[4, :][x[4, :] == 0] = np.nanmean(new)
        x[4, :][x[4, :] < 60] = 60
        x[4, :][x[4, :] > 100] = 100
    
    # Downsample to target frequency
    x_resampled = interp_downsample(x, original_fs, target_freq)
    
    return x_resampled


def parse_annotations(annot_file, event_config, verbose=False):
    """Parse XML annotation file"""
    if verbose:
        print(f"Parsing annotations from: {annot_file}")
    
    try:
        with open(annot_file) as file:
            annot_dict = xmltodict.parse(file.read())
            annot = [x for x in annot_dict['PSGAnnotation']['ScoredEvents']['ScoredEvent']]
    except Exception as e:
        raise ValueError(f"Error parsing annotation file: {e}")
    
    events = []
    
    for annotation in annot:
        name = annotation['EventConcept']
        
        # Handle desaturation events
        if name in event_config['desaturation_names']:
            desat = float(annotation['SpO2Baseline']) - float(annotation['SpO2Nadir'])
            if desat >= event_config['desaturation_threshold']:
                name = 'SpO2 desaturation3'
        
        # Only process relevant events
        if name in (event_config['central_apnea_names'] + 
                   event_config['obstructive_apnea_names'] + 
                   event_config['hypopnea_names'] + 
                   event_config['arousal_names'] + 
                   ['SpO2 desaturation3']):
            
            start = float(annotation['Start'])
            duration = float(annotation['Duration'])
            
            event = {'Event': name, 'StartTime': start, 'Duration': duration}
            events.append(event)
    
    if not events:
        raise ValueError("No relevant events found in annotation file")
    
    return pd.DataFrame(events)


def classify_events(df_event, event_config, verbose=False):
    """Classify and process events"""
    if verbose:
        print("Classifying events...")
    
    # Normalize event names
    for name in event_config['hypopnea_names']:
        df_event.loc[df_event['Event'] == name, 'Event'] = 'Hypopnea|Hypopnea'
    
    for name in event_config['arousal_names']:
        df_event.loc[df_event['Event'] == name, 'Event'] = 'Arousal ()'
    
    # Process hypopnea events (check for associated desaturation/arousal)
    for h in range(len(df_event)):
        if df_event.loc[h, 'Event'] == 'Hypopnea|Hypopnea':
            hyp_end = df_event.loc[h, 'StartTime'] + df_event.loc[h, 'Duration']
            event_end_list = (df_event['StartTime'].values + df_event['Duration'].values)
            event_start_list = df_event['StartTime'].values
            
            # Check for desaturation within 45s after hypopnea
            events_after_desat = df_event[
                (event_end_list < hyp_end + 45) & (event_end_list > hyp_end)
            ]['Event']
            
            # Check for arousal within 5s after hypopnea
            events_after_arousal = df_event[
                (event_start_list < hyp_end + 5) & (event_start_list > hyp_end)
            ]['Event']
            
            if (events_after_desat == 'SpO2 desaturation3').any() or (events_after_arousal == 'Arousal ()').any():
                df_event.loc[h, 'Event'] = 'Hypopnea1'
    
    # Separate event types
    annot_c = df_event[df_event['Event'].isin(event_config['central_apnea_names'])]
    annot_o = df_event[df_event['Event'].isin(event_config['obstructive_apnea_names'])]
    annot_h = df_event[df_event['Event'] == 'Hypopnea1']
    annot_nad_h = df_event[df_event['Event'] == 'Hypopnea|Hypopnea']
    
    return annot_c, annot_o, annot_h, annot_nad_h


def save_processed_data(record_id, x_data, annotations, target_path, target_freq, verbose=False):
    """Save processed data to memory-mapped files"""
    if verbose:
        print(f"Saving processed data for record {record_id}")
    
    # Save signal data
    eeg_filename = os.path.join(target_path, f"{record_id}_eeg.mm")
    signals = np.memmap(eeg_filename, dtype='float32', mode='w+', shape=x_data.shape)
    signals[:, :] = x_data
    
    annotation_dicts = {}
    
    # Process each event type
    event_types = [
        ('OSA', annotations[1], 0),  # Obstructive
        ('CSA', annotations[0], 1),  # Central  
        ('HYPO', annotations[2], 2), # Hypopnea
        ('NAD', annotations[3], 3)   # NAD-Hypopnea
    ]
    
    for event_name, annot_df, label in event_types:
        if annot_df.shape[0] > 0:
            event_filename = os.path.join(target_path, f"{record_id}_{event_name}.mm")
            
            start = annot_df['StartTime'].values.astype(float)
            duration = annot_df['Duration'].values.astype(float)
            
            # Create annotation record
            annotation_record = [
                label, event_name, record_id, eeg_filename, event_filename,
                target_freq, x_data.shape[1], start.shape[0]
            ]
            
            # Save event data
            y = np.concatenate([start.reshape(1, -1), duration.reshape(1, -1)], axis=0)
            labels = np.memmap(event_filename, dtype='float32', mode='w+', shape=y.shape)
            labels[:, :] = y
            
            annotation_dicts[f"{record_id}_{event_name}"] = annotation_record
    
    return annotation_dicts


def main():
    """Main function for command-line execution"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        validate_arguments(args)
        
        if args.verbose:
            print("Starting sleep data preprocessing...")
            print(f"Arguments: {vars(args)}")
        
        # Load input CSV
        df = pd.read_csv(args.input_csv, sep=';')
        if 'edf_f,,' in df.columns:
            df.rename(columns={'edf_f,,': 'edf_f'}, inplace=True)
        
        # Determine processing range
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else df.shape[0]
        
        if args.verbose:
            print(f"Processing records {start_idx} to {end_idx-1}")
        
        # Event configuration
        event_config = {
            'central_apnea_names': args.central_apnea_names,
            'obstructive_apnea_names': args.obstructive_apnea_names,
            'hypopnea_names': args.hypopnea_names,
            'arousal_names': args.arousal_names,
            'desaturation_names': args.desaturation_names,
            'desaturation_threshold': args.desaturation_threshold
        }
        
        # Initialize annotation dictionaries
        df_annotations_o = {}
        df_annotations_c = {}
        df_annotations_h = {}
        df_annotations_nad_h = {}
        
        # Process each record
        for i in tqdm(range(start_idx, end_idx)):
            if args.verbose:
                print(f"\nProcessing record {i}: {df.iloc[i]['f_id']}")
            
            try:
                dfi = df.iloc[i]
                edf_f = dfi.edf_f.replace(',', '')
                annot_f = dfi.annot_f
                record_id = str(dfi.f_id).zfill(4)
                
                # Load arousal data
                arousal, wake = load_arousal_data(args.arousal_path, annot_f, args.target_freq, args.verbose)
                
                # Read EDF file
                raw = read_raw_edf(edf_f, verbose=False)
                
                # Process signal data
                x_processed = process_signal_data(raw, args.channels, args.target_freq, args.verbose)
                
                # Add arousal and wake channels
                arousal_padded = np.concatenate([arousal, np.zeros(x_processed.shape[1] - arousal.shape[0])])
                wake_padded = np.concatenate([wake, np.zeros(x_processed.shape[1] - wake.shape[0])])
                
                x_final = np.concatenate([x_processed, np.stack([arousal_padded, wake_padded])], axis=0)
                
                # Parse annotations
                df_event = parse_annotations(annot_f, event_config, args.verbose)
                
                # Classify events
                annot_c, annot_o, annot_h, annot_nad_h = classify_events(df_event, event_config, args.verbose)
                
                # Save processed data
                annotation_dicts = save_processed_data(
                    record_id, x_final, (annot_c, annot_o, annot_h, annot_nad_h), 
                    args.target_path, args.target_freq, args.verbose
                )
                
                # Update annotation dictionaries
                for key, value in annotation_dicts.items():
                    if '_OSA' in key:
                        df_annotations_o[record_id] = value
                    elif '_CSA' in key:
                        df_annotations_c[record_id] = value
                    elif '_HYPO' in key:
                        df_annotations_h[record_id] = value
                    elif '_NAD' in key:
                        df_annotations_nad_h[record_id] = value
                
            except Exception as e:
                print(f"Error processing record {i}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Save annotation CSVs
        columns = ["label", "event", "record", "eeg_file", "event_file", "fs", "n_times", "n_events"]
        
        for name, annotations in [
            ('OSA', df_annotations_o),
            ('CSA', df_annotations_c), 
            ('HYPO', df_annotations_h),
            ('NAD', df_annotations_nad_h)
        ]:
            if annotations:
                df_annot = pd.DataFrame(annotations).transpose().reset_index(drop=True)
                df_annot.columns = columns
                df_annot.to_csv(os.path.join(args.target_path, f"info_{name}.csv"))
        
        # Save combined CSV
        all_annotations = [df_annotations_o, df_annotations_c, df_annotations_h, df_annotations_nad_h]
        if any(all_annotations):
            combined_data = []
            for annotations in all_annotations:
                combined_data.extend(annotations.values())
            
            if combined_data:
                df_full = pd.DataFrame(combined_data, columns=columns)
                df_full.to_csv(os.path.join(args.target_path, "info_full.csv"))
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Processed files saved to: {args.target_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())