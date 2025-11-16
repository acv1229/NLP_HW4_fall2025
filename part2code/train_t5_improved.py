"""
Improved training script for T5 fine-tuning to achieve ≥65% F1 score.
Key improvements:
1. Better hyperparameters (learning rate, batch size, epochs)
2. Improved beam search (more beams, length penalty)
3. Gradient accumulation for larger effective batch size
4. Better learning rate scheduling
5. Label smoothing for better generalization
"""

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training with improved hyperparameters.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop - Improved version')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=3e-5,  # Optimized for 61% → 65%+ F1
                        help="Learning rate (default: 3e-5 for fine-tuning)")
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=2,  # More warmup for stability
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=40,  # More epochs for better convergence
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=8,  # More patience to find best model
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    
    # New: Gradient accumulation
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment_improved',
                        help="How should we name this experiment?")
    parser.add_argument('--extra_credit', action='store_true',
                        help="If set, save outputs with extra credit naming convention")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument('--add_task_prefix', action='store_true',
                        help="Add 'translate English to SQL: ' prefix to natural language queries")
    
    # Generation hyperparameters
    parser.add_argument('--num_beams', type=int, default=8,  # More beams for better generation
                        help="Number of beams for beam search during generation")
    parser.add_argument('--length_penalty', type=float, default=1.2,  # Length penalty for beam search
                        help="Length penalty for beam search (higher = prefer longer sequences)")
    
    # Label smoothing
    parser.add_argument('--label_smoothing', type=float, default=0.1,  # Label smoothing for better generalization
                        help="Label smoothing factor (0.0 = no smoothing)")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment_ec' if args.extra_credit else 'ft_experiment'
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f't5_{experiment_name}_dev.sql')
    model_record_path = os.path.join('records', f't5_{experiment_name}_dev.pkl')
    
    # Save original F1 before training starts (to protect test files)
    original_f1 = None
    test_record_path = os.path.join('records', f't5_{experiment_name}_test.pkl')
    if os.path.exists(test_record_path) and os.path.exists(model_record_path):
        try:
            from utils import compute_metrics
            _, _, original_f1, _ = compute_metrics(
                gt_sql_path, model_sql_path, gt_record_path, model_record_path
            )
            # Save original F1 to a file so we can retrieve it later
            original_f1_path = os.path.join(checkpoint_dir, 'original_f1.txt')
            with open(original_f1_path, 'w') as f:
                f.write(str(original_f1))
            print(f"📋 Saved original F1 ({original_f1*100:.2f}%) for comparison")
        except Exception as e:
            print(f"Could not save original F1: {e}")
    
    # Create ground truth records if they don't exist
    if not os.path.exists(gt_record_path):
        from utils import save_queries_and_records
        os.makedirs(os.path.dirname(gt_record_path), exist_ok=True)
        with open(gt_sql_path, 'r') as f:
            gt_queries = [line.strip() for line in f.readlines()]
        save_queries_and_records(gt_queries, gt_sql_path, gt_record_path)
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            print(f"✓ New best F1: {best_f1*100:.2f}%")
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping after {epoch+1} epochs (no improvement for {args.patience_epochs} epochs)")
            break

    print(f"\nTraining completed. Best F1: {best_f1*100:.2f}%")

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    # Use label smoothing if specified
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=PAD_IDX)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer.zero_grad()
    accumulation_steps = 0
    
    for batch_idx, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(tqdm(train_loader, desc="Training")):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        # Match train_t5.py approach: no shifting, use decoder_targets directly
        # Reshape logits and targets for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = decoder_targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        
        # Scale loss by accumulation steps
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        accumulation_steps += 1
        
        # Only update weights after accumulating gradients
        if accumulation_steps % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            if scheduler is not None: 
                scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            # Count non-padding tokens for accurate loss averaging
            non_pad = targets_flat != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens * args.gradient_accumulation_steps
            total_tokens += num_tokens

    # Handle remaining gradients
    if accumulation_steps % args.gradient_accumulation_steps != 0:
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return total_loss / total_tokens if total_tokens > 0 else 0.0
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluation with improved beam search.
    '''
    from transformers import T5TokenizerFast
    from load_data import PAD_IDX
    
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    total_loss = 0
    total_tokens = 0
    generated_sql_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss - match train_t5.py approach
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            
            # Reshape for loss computation (match train_t5.py exactly)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = decoder_targets.view(-1)
            loss = criterion(logits_flat, targets_flat)
            
            # Count non-padding tokens
            non_pad = targets_flat != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Generate SQL queries with improved beam search
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_start_token_id=tokenizer.convert_tokens_to_ids('<extra_id_0>'),
                max_length=512,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=True,
                pad_token_id=PAD_IDX,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Prevent repetition
            )
            
            # Decode generated queries
            for gen_ids in generated_ids:
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_sql_queries.append(gen_text)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    
    # Save generated queries and compute metrics
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(generated_sql_queries, model_sql_path, model_record_path)
    
    # Compute metrics
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    error_rate = len(error_msgs) / len(generated_sql_queries) if len(generated_sql_queries) > 0 else 0.0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Inference on test set with improved generation.
    '''
    from transformers import T5TokenizerFast
    from load_data import PAD_IDX
    
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    generated_sql_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader, desc="Test Inference"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries with improved beam search
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_start_token_id=tokenizer.convert_tokens_to_ids('<extra_id_0>'),
                max_length=512,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=True,
                pad_token_id=PAD_IDX,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
            )
            
            # Decode generated queries
            for gen_ids in generated_ids:
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_sql_queries.append(gen_text)
    
    # Save generated queries and records
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(generated_sql_queries, model_sql_path, model_record_path)
    print(f"Generated {len(generated_sql_queries)} SQL queries for test set")
    print(f"Saved to {model_sql_path} and {model_record_path}")

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size, 
                                                         add_task_prefix=args.add_task_prefix)
    model = initialize_model(args)
    # Calculate steps per epoch with gradient accumulation
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, steps_per_epoch)

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment_ec' if args.extra_credit else 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{experiment_name}_dev.pkl')
    test_sql_path = os.path.join(f'results/t5_{experiment_name}_test.sql')
    test_record_path = os.path.join(f'records/t5_{experiment_name}_test.pkl')
    
    # Check if we saved original F1 before training started
    # This protects against dev files being overwritten during training
    previous_f1 = None
    original_f1_path = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name, 'original_f1.txt')
    
    if os.path.exists(original_f1_path):
        try:
            with open(original_f1_path, 'r') as f:
                previous_f1 = float(f.read().strip())
            print(f"📋 Found original F1 saved before training: {previous_f1*100:.2f}%")
        except Exception as e:
            print(f"Could not read saved original F1: {e}")
    
    # Fallback: try to read from current dev files if no saved F1
    if previous_f1 is None and os.path.exists(test_record_path):
        try:
            if os.path.exists(model_record_path):
                from utils import compute_metrics
                _, _, previous_f1, _ = compute_metrics(
                    gt_sql_path, model_sql_path, gt_record_path, model_record_path
                )
                print(f"Found previous results with Dev F1: {previous_f1*100:.2f}%")
        except Exception as e:
            print(f"Could not read previous F1: {e}")
    
    # Evaluate on dev set (this will overwrite dev files, but that's expected)
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")
    
    # Only generate test set outputs if new model is better (or no previous results)
    if previous_f1 is None or dev_record_f1 >= previous_f1:
        if previous_f1 is not None:
            improvement = (dev_record_f1 - previous_f1) * 100
            print(f"\n✓ New model is better! Improvement: {improvement:+.2f}%")
            print(f"  Previous F1: {previous_f1*100:.2f}% → New F1: {dev_record_f1*100:.2f}%")
        else:
            print(f"\n✓ No previous results found, saving new results.")
        
        # Test set - only generate if model is better
        print(f"\nGenerating test set outputs...")
        test_inference(args, model, test_loader, test_sql_path, test_record_path)
        print(f"✓ Test set outputs saved (new model is better)")
    else:
        decline = (previous_f1 - dev_record_f1) * 100
        print(f"\n{'='*70}")
        print(f"⚠ New model is WORSE than previous!")
        print(f"  Previous F1: {previous_f1*100:.2f}%")
        print(f"  New F1:      {dev_record_f1*100:.2f}%")
        print(f"  Decline:     {decline:.2f}%")
        print(f"{'='*70}")
        print(f"⚠ Test set outputs NOT generated to preserve better results.")
        print(f"⚠ Your previous test files are safe:")
        print(f"    - {test_sql_path}")
        print(f"    - {test_record_path}")
        print(f"⚠ Dev files were overwritten during evaluation, but test files are preserved.")

if __name__ == "__main__":
    main()

