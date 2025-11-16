"""
Script to compute data statistics for Q4.
Computes statistics before and after pre-processing using T5 tokenizer.
"""

import os
from transformers import T5TokenizerFast
from collections import Counter
import numpy as np

def load_lines(path):
    """Load lines from a file."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def compute_statistics(nl_path, sql_path, tokenizer, preprocessed=False):
    """
    Compute statistics for a dataset split.
    
    Args:
        nl_path: Path to natural language queries file
        sql_path: Path to SQL queries file
        tokenizer: T5 tokenizer instance
        preprocessed: Whether data has been preprocessed (for Table 2)
    
    Returns:
        Dictionary with statistics
    """
    nl_queries = load_lines(nl_path)
    sql_queries = load_lines(sql_path)
    
    # Number of examples
    num_examples = len(nl_queries)
    assert len(nl_queries) == len(sql_queries), "NL and SQL files must have same length"
    
    # Tokenize and compute lengths
    nl_lengths = []
    sql_lengths = []
    nl_vocab = Counter()
    sql_vocab = Counter()
    
    for nl_query, sql_query in zip(nl_queries, sql_queries):
        # Tokenize natural language query
        nl_tokens = tokenizer.tokenize(nl_query)
        nl_lengths.append(len(nl_tokens))
        nl_vocab.update(nl_tokens)
        
        # Tokenize SQL query
        sql_tokens = tokenizer.tokenize(sql_query)
        sql_lengths.append(len(sql_tokens))
        sql_vocab.update(sql_tokens)
    
    # Compute statistics
    stats = {
        'num_examples': num_examples,
        'mean_nl_length': np.mean(nl_lengths),
        'mean_sql_length': np.mean(sql_lengths),
        'nl_vocab_size': len(nl_vocab),
        'sql_vocab_size': len(sql_vocab),
    }
    
    return stats

def compute_statistics_after_preprocessing(data_folder, split, tokenizer):
    """
    Compute statistics after preprocessing by using the T5Dataset class.
    This assumes you've implemented T5Dataset in load_data.py.
    """
    try:
        from load_data import T5Dataset
        
        # Create dataset instance (this will apply your preprocessing)
        dataset = T5Dataset(data_folder, split)
        
        nl_lengths = []
        sql_lengths = []
        nl_vocab = Counter()
        sql_vocab = Counter()
        
        for i in range(len(dataset)):
            item = dataset[i]
            # Assuming __getitem__ returns (encoder_input_ids, decoder_input_ids, ...)
            # We need to decode to get the tokenized sequences
            # This is a placeholder - adjust based on your actual implementation
            if isinstance(item, tuple) and len(item) >= 2:
                encoder_ids = item[0]
                decoder_ids = item[1] if len(item) > 1 else None
                
                # Decode to get tokens (for vocabulary counting)
                if encoder_ids is not None:
                    encoder_tokens = tokenizer.convert_ids_to_tokens(encoder_ids.tolist())
                    nl_lengths.append(len(encoder_tokens))
                    nl_vocab.update(encoder_tokens)
                
                if decoder_ids is not None:
                    decoder_tokens = tokenizer.convert_ids_to_tokens(decoder_ids.tolist())
                    sql_lengths.append(len(decoder_tokens))
                    sql_vocab.update(decoder_tokens)
        
        stats = {
            'num_examples': len(dataset),
            'mean_nl_length': np.mean(nl_lengths) if nl_lengths else 0,
            'mean_sql_length': np.mean(sql_lengths) if sql_lengths else 0,
            'nl_vocab_size': len(nl_vocab),
            'sql_vocab_size': len(sql_vocab),
        }
        return stats
    except Exception as e:
        print(f"Error computing after-preprocessing stats: {e}")
        print("Make sure you've implemented T5Dataset in load_data.py")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--after_preprocessing', action='store_true',
                       help='Compute statistics after preprocessing (requires T5Dataset implementation)')
    args = parser.parse_args()
    
    # Initialize T5 tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    data_folder = 'data'
    
    print("=" * 80)
    print("TABLE 1: Data statistics BEFORE pre-processing")
    print("=" * 80)
    
    # Compute statistics for train set (before preprocessing)
    train_nl_path = os.path.join(data_folder, 'train.nl')
    train_sql_path = os.path.join(data_folder, 'train.sql')
    train_stats = compute_statistics(train_nl_path, train_sql_path, tokenizer, preprocessed=False)
    
    # Compute statistics for dev set (before preprocessing)
    dev_nl_path = os.path.join(data_folder, 'dev.nl')
    dev_sql_path = os.path.join(data_folder, 'dev.sql')
    dev_stats = compute_statistics(dev_nl_path, dev_sql_path, tokenizer, preprocessed=False)
    
    print("\nStatistics Name | Train | Dev")
    print("-" * 50)
    print(f"Number of examples | {train_stats['num_examples']} | {dev_stats['num_examples']}")
    print(f"Mean sentence length | {train_stats['mean_nl_length']:.2f} | {dev_stats['mean_nl_length']:.2f}")
    print(f"Mean SQL query length | {train_stats['mean_sql_length']:.2f} | {dev_stats['mean_sql_length']:.2f}")
    print(f"Vocabulary size (natural language) | {train_stats['nl_vocab_size']} | {dev_stats['nl_vocab_size']}")
    print(f"Vocabulary size (SQL) | {train_stats['sql_vocab_size']} | {dev_stats['sql_vocab_size']}")
    
    # Save results to a file for easy reference
    with open('statistics_before_preprocessing.txt', 'w') as f:
        f.write("TABLE 1: Data statistics BEFORE pre-processing\n")
        f.write("=" * 80 + "\n\n")
        f.write("Statistics Name | Train | Dev\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of examples | {train_stats['num_examples']} | {dev_stats['num_examples']}\n")
        f.write(f"Mean sentence length | {train_stats['mean_nl_length']:.2f} | {dev_stats['mean_nl_length']:.2f}\n")
        f.write(f"Mean SQL query length | {train_stats['mean_sql_length']:.2f} | {dev_stats['mean_sql_length']:.2f}\n")
        f.write(f"Vocabulary size (natural language) | {train_stats['nl_vocab_size']} | {dev_stats['nl_vocab_size']}\n")
        f.write(f"Vocabulary size (SQL) | {train_stats['sql_vocab_size']} | {dev_stats['sql_vocab_size']}\n")
    
    print("\nResults saved to statistics_before_preprocessing.txt")
    
    # Compute after preprocessing if requested
    if args.after_preprocessing:
        print("\n" + "=" * 80)
        print("TABLE 2: Data statistics AFTER pre-processing")
        print("=" * 80)
        
        train_stats_after = compute_statistics_after_preprocessing(data_folder, 'train', tokenizer)
        dev_stats_after = compute_statistics_after_preprocessing(data_folder, 'dev', tokenizer)
        
        if train_stats_after and dev_stats_after:
            print("\nStatistics Name | Train | Dev")
            print("-" * 50)
            print(f"Mean sentence length | {train_stats_after['mean_nl_length']:.2f} | {dev_stats_after['mean_nl_length']:.2f}")
            print(f"Mean SQL query length | {train_stats_after['mean_sql_length']:.2f} | {dev_stats_after['mean_sql_length']:.2f}")
            print(f"Vocabulary size (natural language) | {train_stats_after['nl_vocab_size']} | {dev_stats_after['nl_vocab_size']}")
            print(f"Vocabulary size (SQL) | {train_stats_after['sql_vocab_size']} | {dev_stats_after['sql_vocab_size']}")
            
            with open('statistics_after_preprocessing.txt', 'w') as f:
                f.write("TABLE 2: Data statistics AFTER pre-processing\n")
                f.write("=" * 80 + "\n\n")
                f.write("Statistics Name | Train | Dev\n")
                f.write("-" * 50 + "\n")
                f.write(f"Mean sentence length | {train_stats_after['mean_nl_length']:.2f} | {dev_stats_after['mean_nl_length']:.2f}\n")
                f.write(f"Mean SQL query length | {train_stats_after['mean_sql_length']:.2f} | {dev_stats_after['mean_sql_length']:.2f}\n")
                f.write(f"Vocabulary size (natural language) | {train_stats_after['nl_vocab_size']} | {dev_stats_after['nl_vocab_size']}\n")
                f.write(f"Vocabulary size (SQL) | {train_stats_after['sql_vocab_size']} | {dev_stats_after['sql_vocab_size']}\n")
            
            print("\nResults saved to statistics_after_preprocessing.txt")
        else:
            print("\nCould not compute after-preprocessing statistics.")
            print("Make sure you've implemented T5Dataset in load_data.py")
    else:
        print("\n" + "=" * 80)
        print("TABLE 2: Data statistics AFTER pre-processing")
        print("=" * 80)
        print("\nNOTE: To compute statistics after pre-processing:")
        print("1. Implement your data processing in load_data.py (T5Dataset class)")
        print("2. Run: python3 compute_statistics.py --after_preprocessing")
        print("\nIf you don't do any pre-processing, the statistics will be identical to Table 1.")
        print("If you do pre-processing (e.g., adding prefixes like 'translate English to SQL:',")
        print("special tokens, etc.), the statistics may change.")

if __name__ == "__main__":
    main()

