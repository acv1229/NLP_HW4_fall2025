import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, add_task_prefix=True):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        
        Args:
            add_task_prefix: If True, prepends "translate English to SQL: " to natural language queries.
                            This can help training from scratch by giving the model task context.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.add_task_prefix = add_task_prefix
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')
        
        nl_queries = load_lines(nl_path)
        
        # For test set, we don't have SQL queries
        if split == 'test':
            sql_queries = [None] * len(nl_queries)
        else:
            sql_queries = load_lines(sql_path)
        
        assert len(nl_queries) == len(sql_queries), f"Mismatch in data lengths: {len(nl_queries)} vs {len(sql_queries)}"
        
        return list(zip(nl_queries, sql_queries))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nl_query, sql_query = self.data[idx]
        
        # Optionally add task prefix to help model understand the task (useful for training from scratch)
        if self.add_task_prefix:
            nl_query = f"translate English to SQL: {nl_query}"
        
        # Tokenize encoder input (natural language query)
        encoder_inputs = self.tokenizer(
            nl_query,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        encoder_ids = encoder_inputs['input_ids']
        
        if self.split == 'test':
            # For test set, we only return encoder inputs
            decoder_start_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
            return {
                'encoder_ids': torch.tensor(encoder_ids, dtype=torch.long),
                'initial_decoder_input': torch.tensor([decoder_start_token_id], dtype=torch.long)
            }
        
        # Tokenize decoder input (SQL query)
        # Add decoder start token (extra_id_0) at the beginning
        decoder_inputs = self.tokenizer(
            sql_query,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        sql_ids = decoder_inputs['input_ids']
        
        # Decoder input: prepend extra_id_0 (decoder start token)
        decoder_start_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        decoder_input_ids = [decoder_start_token_id] + sql_ids
        
        # Decoder targets: shift by one position (next token prediction)
        # Target should be the SQL tokens (what we want to predict)
        # We'll use -100 for tokens we don't want to compute loss on (like padding)
        decoder_target_ids = sql_ids + [self.tokenizer.eos_token_id]
        
        return {
            'encoder_ids': torch.tensor(encoder_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_target_ids': torch.tensor(decoder_target_ids, dtype=torch.long),
            'initial_decoder_input': torch.tensor([decoder_start_token_id], dtype=torch.long)
        }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]
    decoder_target_ids_list = [item['decoder_target_ids'] for item in batch]
    initial_decoder_inputs_list = [item['initial_decoder_input'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_target_ids_list, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs_list).squeeze(1)
    
    # Create attention masks
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    initial_decoder_inputs_list = [item['initial_decoder_input'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs_list).squeeze(1)
    
    # Create attention masks
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, add_task_prefix=True):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, add_task_prefix=add_task_prefix)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, add_task_prefix=True):
    train_loader = get_dataloader(batch_size, "train", add_task_prefix=add_task_prefix)
    dev_loader = get_dataloader(test_batch_size, "dev", add_task_prefix=add_task_prefix)
    test_loader = get_dataloader(test_batch_size, "test", add_task_prefix=add_task_prefix)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x