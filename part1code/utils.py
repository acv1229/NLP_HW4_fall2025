import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Synonym replacement transformation
    # Randomly replace words with their synonyms to create OOD data
    # This is reasonable because users might use different words with similar meanings
    
    text = example["text"]
    words = word_tokenize(text)
    detokenizer = TreebankWordDetokenizer()
    
    # Probability of replacing a word with a synonym
    replace_prob = 0.5
    
    transformed_words = []
    for word in words:
        # Only replace with some probability and if wordnet has synonyms
        if random.random() < replace_prob:
            # Get synsets (synonym sets) for the word
            synsets = wordnet.synsets(word)
            if synsets:
                # Get all synonyms from all synsets
                synonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        # Don't use the same word and prefer lowercase
                        if synonym.lower() != word.lower():
                            synonyms.append(synonym.lower())
                
                # If we found synonyms, randomly pick one
                if synonyms:
                    # Remove duplicates while preserving order
                    unique_synonyms = list(dict.fromkeys(synonyms))
                    word = random.choice(unique_synonyms)
        
        transformed_words.append(word)
    
    # Reconstruct the text
    example["text"] = detokenizer.detokenize(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example
