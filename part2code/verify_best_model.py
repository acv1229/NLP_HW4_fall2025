#!/usr/bin/env python3
"""
Quick script to verify the validation F1 score for the best saved model.
"""

import os
import sys
import argparse
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from t5_utils import load_model_from_checkpoint
from load_data import load_t5_data
from train_t5 import eval_epoch
from utils import save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    """Create args object matching training configuration"""
    parser = argparse.ArgumentParser(description='Verify best model validation F1')
    parser.add_argument('--experiment_name', type=str, default='ft_experiment_prefix',
                        help='Experiment name (must match training)')
    parser.add_argument('--add_task_prefix', action='store_true', default=True,
                        help='Whether task prefix was used during training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    return parser.parse_args()

def main():
    # Create args object
    class Args:
        def __init__(self, parsed_args):
            self.finetune = True
            self.experiment_name = parsed_args.experiment_name
            self.add_task_prefix = parsed_args.add_task_prefix
            self.batch_size = parsed_args.batch_size
            self.test_batch_size = parsed_args.test_batch_size
            self.learning_rate = 5e-5
            self.max_n_epochs = 30
            self.patience_epochs = 5
            self.scheduler_type = 'cosine'
            self.num_warmup_epochs = 1
            self.weight_decay = 0.01
            self.optimizer_type = 'AdamW'
            self.use_wandb = False
            self.extra_credit = False
    
    parsed_args = get_args()
    args = Args(parsed_args)
    
    print("="*70)
    print("VERIFYING BEST MODEL VALIDATION F1")
    print("="*70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Task prefix: {args.add_task_prefix}")
    print(f"Device: {DEVICE}")
    print("="*70)
    
    # Check if checkpoint exists
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n✓ Found checkpoint: {checkpoint_path}")
    
    # Load the best model
    print("\nLoading best model...")
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load data
    print("\nLoading data...")
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size,
        args.test_batch_size,
        add_task_prefix=args.add_task_prefix
    )
    print(f"✓ Data loaded (dev batches: {len(dev_loader)})")
    
    # Set up paths
    experiment_name = 'ft_experiment_ec' if args.extra_credit else 'ft_experiment'
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f't5_{experiment_name}_dev_verify.sql')
    model_record_path = os.path.join('records', f't5_{experiment_name}_dev_verify.pkl')
    
    # Create ground truth records if needed
    if not os.path.exists(gt_record_path):
        os.makedirs(os.path.dirname(gt_record_path), exist_ok=True)
        with open(gt_sql_path, 'r') as f:
            gt_queries = [line.strip() for line in f.readlines()]
        save_queries_and_records(gt_queries, gt_sql_path, gt_record_path)
        print(f"Created ground truth records: {gt_record_path}")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING ON DEV SET")
    print("="*70)
    
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    
    # Print results
    print("\n" + "="*70)
    print("DEV SET RESULTS (BEST MODEL)")
    print("="*70)
    print(f"Loss:              {dev_loss:.6f}")
    print(f"Record F1:         {dev_record_f1*100:.2f}%")
    print(f"Record EM:         {dev_record_em*100:.2f}%")
    print(f"SQL EM:            {dev_sql_em*100:.2f}%")
    print(f"SQL Error Rate:    {dev_error_rate*100:.2f}%")
    print("="*70)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Generated SQL saved to: {model_sql_path}")
    print(f"  Generated records saved to: {model_record_path}")

if __name__ == "__main__":
    main()





