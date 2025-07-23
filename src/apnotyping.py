#!/usr/bin/env python3
"""
Apnotyping Data Generation Script

Command-line tool for generating apnotyping probability files from sleep event detection model

Usage:
    python apnotyping.py --data_input /path/to/data.csv --model_path /path/to/model.pth --output_dir /path/to/output/

@author: Magnus Ruud Kjær
Updated: 2025
"""

import pandas as pd
import numpy as np
import os
import argparse
import sys
import pickle
from tqdm import tqdm
import torch

# Custom imports
from datagenerator.datagen import DataGen


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sleep Event Detection Apnotyping Data Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python apnotyping.py --data_input /data/sleep_data.csv --model_path /models/saved_model.pth --output_dir /output/apnotypes/
    
    python apnotyping.py --data_input /data/sleep_data.csv --model_path /models/saved_model.pth --output_dir /output/apnotypes/ --test_size 500
        """
    )
    
    parser.add_argument(
        '--data_input', 
        type=str, 
        required=True,
        help='Path to CSV file containing the dataset'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to saved model (.pth file)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Directory to save apnotyping pickle files'
    )
    
    parser.add_argument(
        '--test_size', 
        type=int, 
        default=None,
        help='Number of test cases to process (default: process all records)'
    )
    
    parser.add_argument(
        '--window', 
        type=int, 
        default=240,
        help='Window size for analysis (default: 240)'
    )
    
    parser.add_argument(
        '--detection_threshold', 
        type=float, 
        default=0.51,
        help='Detection threshold for apnotyping (default: 0.51)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    # Check data input file exists
    if not os.path.exists(args.data_input):
        raise FileNotFoundError(f"Data file not found: {args.data_input}")
    
    # Check model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Check/create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Check test size is positive if specified
    if args.test_size is not None and args.test_size <= 0:
        raise ValueError("test_size must be positive")
    
    # Check detection threshold
    if args.detection_threshold < 0 or args.detection_threshold > 1:
        raise ValueError("detection_threshold must be between 0 and 1")


def load_and_prepare_data(data_path, test_size=None, verbose=False):
    """
    Load and prepare data from CSV file
    
    Args:
        data_path: Path to CSV file
        test_size: Number of test cases to process (None for all)
        verbose: Whether to print detailed info
    
    Returns:
        tuple: (full_df, test_df, test_records)
    """
    if verbose:
        print(f"Loading data from: {data_path}")
    
    # Load individual datasets (mimicking original structure)
    try:
        df = pd.read_csv(data_path, delimiter=',')
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if 'record' not in df.columns:
        raise ValueError("DataFrame must contain 'record' column")
    
    # Filter records (mimicking original filtering)
    df = df[np.isin(df.record.values, df.record.unique())]
    
    # Determine test records
    unique_records = sorted(pd.unique(df.record))
    
    if test_size is not None:
        if len(unique_records) < test_size:
            raise ValueError(f"Not enough records. Available: {len(unique_records)}, Requested: {test_size}")
        test_records = unique_records[:test_size]
    else:
        test_records = unique_records
    
    df_test = df[df.record.isin(test_records)]
    
    if verbose:
        print(f"Total records available: {len(unique_records)}")
        print(f"Processing records: {len(test_records)}")
    
    return df, df_test, test_records


def load_model(model_path, verbose=False):
    """
    Load the saved model
    
    Args:
        model_path: Path to saved model
        verbose: Whether to print detailed info
    
    Returns:
        torch.nn.Module: Loaded model
    """
    if verbose:
        print(f"Loading model from: {model_path}")
    
    # Load complete model (not just state dict)
    model = torch.load(model_path)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if verbose:
        print(f"Model loaded successfully")
        print(f"Using device: {device}")
    
    return model


def generate_apnotyping_files(model, df, df_test, test_records, config, output_dir, verbose=False):
    """
    Generate apnotyping files for all test records
    
    Args:
        model: Loaded model
        df: Full DataFrame
        df_test: Test DataFrame
        test_records: List of test record IDs
        config: Configuration dictionary
        output_dir: Output directory for pickle files
        verbose: Whether to print detailed info
    
    Returns:
        int: Number of files generated
    """
    print(f"Generating apnotyping files for {len(test_records)} records...")
    
    progress_bar = tqdm(test_records) if not verbose else test_records
    files_generated = 0
    
    for i, record in enumerate(progress_bar):
        if verbose:
            print(f"Processing record {i+1}/{len(test_records)}: {record}")
        
        try:
            # Generate predictions for this record
            pred_gen = DataGen(
                df[df.record == record], 
                config['selected_channels'],
                config['window'], 
                config['downsampling'],
                config['minimum_overlap'],
                config['channels'],
                index_on_events=False, 
                ratio_positive=None
            )
            
            # Get predictions with probability data
            y_true, y_pred, y_prob, events_df_prob = model.predict_generator(
                pred_gen, 
                config['detection_threshold']
            )
            
            # Save apnotyping data to pickle file
            output_file = os.path.join(output_dir, f'events_df_prob_{record}.pkl')
            
            with open(output_file, 'wb') as f:
                pickle.dump((y_pred, y_prob, y_true, events_df_prob), f)
            
            files_generated += 1
            
            if verbose:
                print(f"  Saved: {output_file}")
                
        except Exception as e:
            print(f"Error processing record {record}: {e}")
            continue
    
    print(f"Successfully generated {files_generated} apnotyping files")
    return files_generated


def main():
    """Main function for command-line execution"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        validate_arguments(args)
        
        if args.verbose:
            print("Starting apnotyping data generation...")
            print(f"Arguments: {vars(args)}")
        
        # Configuration
        config = {
            'selected_channels': ['Flow', 'Therm', 'Thor', 'Abdo', 'SpO2', 'Arousal', 'Wake'],
            'channels': ['Flow', 'Therm', 'Thor', 'Abdo', 'SpO2', 'Arousal', 'Wake'],
            'window': args.window,
            'downsampling': 1,
            'minimum_overlap': 0.5,
            'detection_threshold': [args.detection_threshold]  # Single threshold for apnotyping
        }
        
        # Load and prepare data
        df, df_test, test_records = load_and_prepare_data(
            args.data_input, args.test_size, args.verbose
        )
        
        # Load model
        model = load_model(args.model_path, args.verbose)
        
        # Generate apnotyping files
        files_generated = generate_apnotyping_files(
            model, df, df_test, test_records, config, args.output_dir, args.verbose
        )
        
        print(f"\nApnotyping data generation completed successfully!")
        print(f"Generated {files_generated} files in: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())