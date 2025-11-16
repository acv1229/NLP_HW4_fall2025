"""
Script to evaluate the previous best model (57.6% F1) with beam search.
This loads the checkpoint from the first training run and evaluates it.
"""

import os
import sys

# Add current directory to path (works in both local and Colab)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
from train_t5 import get_args, load_t5_data, eval_epoch, test_inference
from t5_utils import load_model_from_checkpoint

def main():
    # Create args object matching the original training run
    args = argparse.Namespace()
    args.finetune = True
    args.experiment_name = 'ft_experiment'  # Original experiment name
    args.batch_size = 16
    args.test_batch_size = 16
    
    print("Loading the best model from first training run...")
    print(f"Looking for checkpoint: checkpoints/ft_experiments/{args.experiment_name}/best_model.pt")
    
    # Load the best model from the first run
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load data
    print("Loading data...")
    _, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    
    # Evaluate on dev set
    print("\n" + "="*80)
    print("Evaluating on DEV set with beam search (num_beams=4)...")
    print("="*80)
    
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', 't5_ft_ft_experiment_dev_beam4.sql')
    model_record_path = os.path.join('records', 't5_ft_ft_experiment_dev_beam4.pkl')
    
    # Create ground truth records if they don't exist
    if not os.path.exists(gt_record_path):
        from utils import save_queries_and_records
        os.makedirs(os.path.dirname(gt_record_path), exist_ok=True)
        with open(gt_sql_path, 'r') as f:
            gt_queries = [line.strip() for line in f.readlines()]
        save_queries_and_records(gt_queries, gt_sql_path, gt_record_path)
    
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    
    print("\n" + "="*80)
    print("DEV SET RESULTS (with beam search):")
    print("="*80)
    print(f"Loss: {dev_loss:.4f}")
    print(f"Record F1: {dev_record_f1:.4f} ({dev_record_f1*100:.2f}%)")
    print(f"Record EM: {dev_record_em:.4f} ({dev_record_em*100:.2f}%)")
    print(f"SQL EM: {dev_sql_em:.4f} ({dev_sql_em*100:.2f}%)")
    print(f"Error Rate: {dev_error_rate*100:.2f}%")
    print("="*80)
    
    # Generate test set predictions
    print("\n" + "="*80)
    print("Generating TEST set predictions with beam search...")
    print("="*80)
    
    model_sql_path = os.path.join('results', 't5_ft_ft_experiment_test_beam4.sql')
    model_record_path = os.path.join('records', 't5_ft_ft_experiment_test_beam4.pkl')
    
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print(f"Dev results saved to: {model_sql_path.replace('test', 'dev')}")
    print(f"Test results saved to: {model_sql_path}")
    print("\nIf F1 improved, you can submit the beam4 files!")

if __name__ == "__main__":
    main()

