"""
Sleep Event Detection Model Evaluation Script

Runs trained model and evaluates on specified cohort

@author: Magnus Ruud Kjær
Updated: 2025
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

# Custom imports
from datagenerator.datagen import DataGen
from model.event_detector import EventDetector
from metrics.metrics import multi_evaluation


def load_and_prepare_data(data_input, test_size=1000):
    """
    Load and prepare data from either DataFrame or CSV file path
    
    Args:
        data_input: Either pandas DataFrame or path to CSV file
        test_size: Number of test cases to extract
    
    Returns:
        tuple: (full_df, test_df, train_records, val_records, test_records)
    """
    # Handle data input
    if isinstance(data_input, str):
        assert os.path.exists(data_input), f"CSV file not found: {data_input}"
        df = pd.read_csv(data_input, delimiter=',')
        print(f"Loaded data from: {data_input}")
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
        print("Using provided DataFrame")
    else:
        raise ValueError("data_input must be either a DataFrame or path to CSV file")
    
    assert not df.empty, "DataFrame is empty"
    assert 'record' in df.columns, "DataFrame must contain 'record' column"
    
    # Prepare test data
    unique_records = sorted(pd.unique(df.record))
    assert len(unique_records) >= test_size, f"Not enough records. Available: {len(unique_records)}, Requested: {test_size}"
    
    # Take first test_size records for testing
    test_records = unique_records[:test_size]
    df_test = df[df.record.isin(test_records)]
    
    # Split remaining records for train/val
    remaining_records = [r for r in unique_records if r not in test_records]
    
    if len(remaining_records) > 0:
        train_records, val_records = train_test_split(
            remaining_records, 
            train_size=0.9, 
            test_size=0.1, 
            random_state=30
        )
    else:
        train_records, val_records = [], []
    
    print(f"Total records: {len(unique_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    
    return df, df_test, train_records, val_records, test_records


def load_model(model_path, model_config):
    """
    Load and initialize the trained model
    
    Args:
        model_path: Path to saved model weights
        model_config: Dictionary containing model configuration
    
    Returns:
        torch.nn.Module: Loaded model
    """
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    
    # Initialize model
    model = EventDetector(**model_config)
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model loaded from: {model_path}")
    print(f"Using device: {device}")
    
    return model


def evaluate_records(model, df_test, eval_config, output_path=None):
    """
    Evaluate model on test records
    
    Args:
        model: Trained model
        df_test: Test DataFrame
        eval_config: Dictionary containing evaluation configuration
        output_path: Path to save results CSV (optional)
    
    Returns:
        pandas.DataFrame: Evaluation results
    """
    records = sorted(pd.unique(df_test.record))
    
    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=[
        'record', 'op_ahi_0', 'precision_0', 'recall_0', 'f1_0', 'n_match_0', 
        'n_pred_0', 'n_true_0', 'avg_iou_0', 'std_iou_0', 'op_ahi_1', 
        'precision_1', 'recall_1', 'f1_1', 'n_match_1', 'n_pred_1', 
        'n_true_1', 'avg_iou_1', 'std_iou_1', 'op_ahi_2', 'precision_2', 
        'recall_2', 'f1_2', 'n_match_2', 'n_pred_2', 'n_true_2', 
        'avg_iou_2', 'std_iou_2', 'n_match_all', 'n_pred_all', 'n_true_all',
        'n_match_0a', 'n_match_1a', 'n_match_2a', 'n_match_01', 'n_match_02',
        'n_match_10', 'n_match_12', 'n_match_20', 'n_match_21'
    ])
    
    print(f"Evaluating {len(records)} records...")
    
    for i, record in enumerate(tqdm(records)):
        # Generate predictions for this record
        pred_gen = DataGen(
            df_test[df_test.record == record], 
            eval_config['selected_channels'],
            eval_config['window'], 
            eval_config['downsampling'],
            eval_config['minimum_overlap'],
            eval_config['channels'],
            index_on_events=False, 
            ratio_positive=None
        )
        
        # Get predictions
        y_true, y_pred, y_prob = model.predict_generator(
            pred_gen, 
            eval_config['detection_threshold']
        )
        
        # Calculate probability distributions for each event type
        prob_distributions = calculate_probability_distributions(y_pred, y_prob)
        
        # Post-process predictions
        y_pred = post_process_predictions(y_pred)
        
        # Evaluate performance
        evaluation_results = multi_evaluation(
            y_true.squeeze(), 
            y_pred.squeeze(),
            sfreq=pred_gen.fs, 
            iou_ths=[0.0001]
        )
        
        # Add record ID and probability distributions
        evaluation_results["record"] = record
        for event_type, dist in prob_distributions.items():
            for j, prob in enumerate(dist):
                evaluation_results[f"dist_{event_type}_{j-1}"] = prob
        
        # Append to results
        results_df = pd.concat([results_df, pd.DataFrame([evaluation_results])], ignore_index=True)
    
    # Save results if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        output_path = "evaluation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to default: {output_path}")
    
    return results_df


def calculate_probability_distributions(y_pred, y_prob):
    """Calculate probability distributions for each event type"""
    distributions = {}
    
    for event_type in range(4):  # OA, CA, HYPO, NAD
        event_indices = np.where(y_pred[event_type, event_type, :] == 1)[0]
        
        if event_indices.size > 0:
            # Calculate mean probability distribution for this event type
            prob_dist = (np.mean(y_prob[0, :, event_indices], axis=0) + 
                        np.mean(y_prob[1, :, event_indices], axis=0)) / 2
        else:
            prob_dist = np.array([0, 0, 0, 0, 0])
        
        distributions[event_type] = prob_dist
    
    return distributions


def post_process_predictions(y_pred):
    """Post-process predictions to handle overlaps"""
    # Copy predictions to first dimension
    y_pred[0, 1, :] = y_pred[1, 1, :]
    y_pred[0, 2, :] = y_pred[2, 2, :]
    y_pred[0, 3, :] = y_pred[3, 3, :]
    
    # Remove overlaps
    y_pred[0, 0, :] = y_pred[0, 0, :] - np.multiply(y_pred[0, 0, :], y_pred[0, 1, :])
    y_pred[0, 3, :] = (y_pred[0, 3, :] - 
                      np.multiply(y_pred[0, 3, :], y_pred[0, 1, :]) - 
                      np.multiply(y_pred[0, 3, :], y_pred[0, 0, :]))
    y_pred[0, 2, :] = (y_pred[0, 2, :] - 
                      np.multiply(y_pred[0, 2, :], y_pred[0, 1, :]) - 
                      np.multiply(y_pred[0, 2, :], y_pred[0, 0, :]) - 
                      np.multiply(y_pred[0, 2, :], y_pred[0, 3, :]))
    
    return y_pred[0, :, :]


def main(data_input, test_size, model_path, output_path=None):
    """
    Main evaluation function
    
    Args:
        data_input: Either pandas DataFrame or path to CSV file
        test_size: Number of test cases to extract
        model_path: Path to saved model weights
        output_path: Path to save results CSV (optional)
    
    Returns:
        pandas.DataFrame: Evaluation results
    """
    # Assertions for required inputs
    assert data_input is not None, "data_input must be specified"
    assert test_size > 0, "test_size must be positive"
    assert model_path is not None, "model_path must be specified"
    
    # Configuration
    eval_config = {
        'selected_channels': ['Flow', 'Therm', 'Thor', 'Abdo', 'SpO2', 'Arousal', 'Wake'],
        'channels': ['Flow', 'Therm', 'Thor', 'Abdo', 'SpO2', 'Arousal', 'Wake'],
        'window': 240,
        'downsampling': 1,
        'minimum_overlap': 0.5,
        'detection_threshold': [0.51, 0.71, 0.61, 0.71]
    }
    
    # Model configuration
    model_config = {
        'n_channels': len(eval_config['selected_channels']),
        'n_classes': 4,
        'n_freq_channels': 0,
        'freq_factor': [1, 1, 1, 1, 1, 1, 1],
        'n_times': eval_config['window'] * 8,  # Assuming 8 Hz sampling rate
        'num_workers': 10,
        'fs': 8,
        'histories_path': '',  # Not needed for evaluation
        'weights_path': '',    # Not needed for evaluation
        'loss': "worst_negative_mining",
        'default_event_sizes': [80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960],
        'factor_overlap': 2,
        'lr': 0.001,
        'epochs': 8,
        'k_max': 5,
        'max_pooling': 2,
        'batch_size': 128,
        'dropout': 0.9,
        'partial_eval': -1,
        'linearlayer': 0,
        'RES_architecture': [3, 4, 6, 3]
    }
    
    # Load and prepare data
    df, df_test, train_records, val_records, test_records = load_and_prepare_data(
        data_input, test_size
    )
    
    # Load model
    model = load_model(model_path, model_config)
    
    # Evaluate model
    results = evaluate_records(model, df_test, eval_config, output_path)
    
    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    for event_type in range(3):  # OA, CA, HYPO
        precision = results[f'precision_{event_type}'].mean()
        recall = results[f'recall_{event_type}'].mean()
        f1 = results[f'f1_{event_type}'].mean()
        print(f"Event {event_type}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    # You need to specify these inputs:
    
    # Input 1: Data source (DataFrame or CSV path)
    data_input = None  # MUST BE SPECIFIED
    
    # Input 2: Number of test cases
    test_size = None   # MUST BE SPECIFIED
    
    # Input 3: Model path
    model_path = None  # MUST BE SPECIFIED
    
    # Optional: Output path for results
    output_path = None  # Optional
    
    # Run evaluation
    try:
        results = main(data_input, test_size, model_path, output_path)
        print("Evaluation completed successfully!")
    except AssertionError as e:
        print(f"Configuration error: {e}")
        print("\nPlease specify the required inputs:")
        print("- data_input: Path to CSV file or pandas DataFrame")
        print("- test_size: Number of test cases to extract")
        print("- model_path: Path to saved model weights (.pth file)")
    except Exception as e:
        print(f"Error during evaluation: {e}")