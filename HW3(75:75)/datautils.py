"""
Utility functions for reading in the dataset and creating 
Dataset and DataLoader objects 
"""
from collections import defaultdict
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad

from model import autoregressive_mask

# constants
VOCAB_DIR = "vocab"
SRC_LANG = "de"
TGT_LANG = "en"

# special tokens
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"  # padding
UNK = "<unk>"

BOS_ID = 0
EOS_ID = 1
PAD_ID = 2
UNK_ID = 3


def convert_and_join(int_to_str_dict, list_of_lists):
    converted_list_of_lists = [[int_to_str_dict[num] for num in sublist] for sublist in list_of_lists]
    out = [' '.join(sublist) for sublist in converted_list_of_lists]
    # Remove special tokens
    out = [s.replace(BOS_WORD, '').replace(EOS_WORD, '').replace(BLANK_WORD, '').strip() for s in out]
    return out


def get_special_tokens():
    """
    Get the special tokens for the vocabulary. This is for initializing vocabularies. 
    """
    vocab = {
        BOS_WORD: BOS_ID,
        EOS_WORD: EOS_ID,
        BLANK_WORD: PAD_ID,  # padding 
        UNK: UNK_ID,
    }
    return vocab


def get_pad_id():
    """
    Get the padding id. This function is for exporting PAD_ID to other files. 
    """
    return PAD_ID


# dataset 
class TranslationDataset(Dataset):
    """ 
    Dataset object for translation data. 

    Input Params: 
        - data (list): list of source and target data. 
        - src_vocab (dict): vocabulary for source data. 
        - tgt_vocab (dict): vocabulary for target data. 
    """

    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the source and target data for a given index. 
        Input Params: 
            - idx (int): index of the data to get. 
        Returns: 
            - src_ids (list): list of source ids. 
            - tgt_ids (list): list of target ids (if not test set). 
        """
        src, tgt = self.data[idx]
        src_ids = [self.src_vocab[token] for token in src]
        tgt_ids = [self.tgt_vocab[token] for token in tgt]
        
        return src_ids, tgt_ids


# data / pretokenization 
def read_pairs(src_path, tgt_path, test_set=False):
    """ 
    Read in the source and target data, pretokenize them, and return a list of source and target sentences. 

    Input Params: - src_path (str): path to source data. - tgt_path (str): path to target data. Returns: - data (
    List[Tuple[List[str], List[str]]]): list of source and target data. Each element is a pair of sentences in the
    source and target languages. The sentences are pretokenized, i.e. is a list of strings obtained from splitting
    the original text at whitespaces.
    """
    # read in data 
    src = open(src_path, encoding="utf-8", errors="replace").readlines()
    tgt = None
    if not test_set and tgt_path is not None:
        tgt = open(tgt_path, encoding="utf-8", errors="replace").readlines()
    else:
        tgt = ['dummy target'] * len(src)
    
    assert len(src) == len(tgt), "Source and target data must have the same number of lines."
    # pretokenize
    data = []
    for src_line, tgt_line in zip(src, tgt):
        src_sent = src_line.strip().split()
        tgt_sent = tgt_line.strip().split()
        data.append((src_sent, tgt_sent))
    return data


# vocab 
def build_vocab(sentences, lang):
    """
    Build a vocabulary from a list of sentences. Since the tokenization process might take some time, we save the
    resulting vocabularies to a file.

    Input Params: 
        - sentences (list): list of sentences to build vocabulary from. 
        - tokenizer (function): tokenizer to 
        - lang (str): language of the sentences. (e.g. "de" or "en") This is just for naming the saved file. 
    Returns: 
        - vocab (dict): vocabulary. 
    """
    assert type(sentences[0]) == list, f"sentences is a list of list of tokens contained in each sentence."

    # initialize vocab with special tokens 
    vocab = get_special_tokens()
    vocab = defaultdict(lambda: UNK_ID, vocab)

    # check local for vocabulary file 
    vocab_path = f"{VOCAB_DIR}/{lang}_tokens.json"
    os.makedirs(VOCAB_DIR, exist_ok=True)

    if os.path.exists(vocab_path):
        print(f"Loading {lang} tokens from local file...")
        with open(vocab_path, "r") as f:
            vocab.update(json.load(f))
        return vocab
    else:
        print(f"Building {lang} tokens...")

        # tokenize sentences 
        unique_tokens = set()
        for sent in sentences:
            for tokens in sent:
                for token in sent:
                    if isinstance(token, str):
                        unique_tokens.add(token)
                    else:
                        # If token is not a string, convert it to a string
                        unique_tokens.add(str(token))

        # add tokens to vocab 
        for token in unique_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        print(f"len of vocab: {len(vocab)}")

        # save vocab 
        with open(vocab_path, "w") as f:
            json.dump(dict(vocab), f, indent=4)  # Convert defaultdict to regular dict
        return vocab


def get_vocab_size():
    """
    Get the vocabulary size for the given source and target data files: 
    train.de-en.{de/en} and valid.de-en.{de/en}. This is so that we don't have to rebuild 
    the vocab when initializing model architectures which depend on the vocab size. 
    """
    src_vocab = 103788
    tgt_vocab = 49936
    return src_vocab, tgt_vocab


# batching
class Batch:
    """
    Object for holding a batch of data for the dataloader. 
    This adds masking to the target data. 

    Input Params: 
        - src (Tensor): source data. Shape: (batch_size, seq_len)
        - tgt (Tensor): target data. Shape: (batch_size, seq_len)
        - pad (int): padding id. 
    """

    def __init__(self, src, tgt=None, pad=PAD_ID):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # ignore bos 
            self.tgt = tgt[:, :-1]

            # ignore eos 
            self.tgt_y = tgt[:, 1:]

            # mask padding and future words 
            self.tgt_mask = self.make_std_mask(self.tgt, pad)

            # number of tokens in target 
            self.ntokens = (self.tgt_y != pad).data.sum()

    def __repr__(self):
        return f"Batch(src={self.src}, tgt={self.tgt}, src_mask={self.src_mask}, tgt_mask={self.tgt_mask})"

    def get_token_strings(self, src_vocab, tgt_vocab):
        """
        Get the token strings for the source and target data. 
        """
        self.src_tokens = [src_vocab[token] for token in self.src]
        self.tgt_tokens = [tgt_vocab[token] for token in self.tgt]
        return self.src_tokens, self.tgt_tokens

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        Input Params: 
            - tgt (Tensor): target data. 
            - pad (int): padding id. 
        Returns: 
            - tgt_mask (Tensor): mask to hide padding and future words.

        HINTS:  
        1. Create a mask to hide padding in the target data. 
        2. Create an autogregressive mask to hide future words in the target data. 
        (Use the autoregressive_mask function in model.py, Section 2.8. of notebook)
        3. Combine the padding mask and the autoregressive mask. 
        """
        # TODO: Implement the make_std_mask function.
        # YOUR CODE STARTS HERE
        B, T = tgt.size(0), tgt.size(1)

        # causal (autoregressive) mask: [1, T, T] with True on allowed positions (<= current)
        ar_mask = autoregressive_mask(T).to(tgt.device)  # no device kwarg; move with .to(...)

        # key-side padding mask broadcast across query positions → [B, T, T]
        # (mask out columns where key is PAD)
        key_keep = (tgt != pad).unsqueeze(1).expand(B, T, T)

        # combine to [B, T, T]
        tgt_mask = key_keep & ar_mask  # ar_mask broadcasts over batch
        # YOUR CODE ENDS HERE
        return tgt_mask


def collate_batch(batch_list,
                  device,
                  max_padding=128):
    """
    Collate and process a batch of source and target sentences for model input.

    This function takes a list of sentence pairs (source and target), processes them by:
    1. Adding special tokens for the beginning and end of sentences (e.g., <s>, </s>).
    2. Padding the sentences to a uniform length (`max_padding`), ensuring that all 
       sentences in the batch have the same length for easier batching in training. 
       Sentences shorter than `max_padding` are padded with a designated padding token (PAD_ID).
       Sentences longer than `max_padding` are truncated (handled implicitly).
       
    Parameters:
    batch_list : List[Tuple[List[int], List[int]]]
        A list of tuples, where each tuple contains a source sentence and a target sentence, 
        represented as lists of token IDs.
    
    device : torch.device
        The device (CPU or GPU) on which the tensors will be allocated.
    
    max_padding : int, optional, default=128
        The maximum length to which the sentences will be padded. All sentences shorter than
        this length will be padded to this size, and all sentences longer than this will 
        be truncated to fit.

    Returns:
    src : torch.Tensor
        A tensor of size (batch_size, max_padding) containing the padded source sentences,
        where each sentence has special tokens <s> and </s> added and padding applied.

    tgt : torch.Tensor
        A tensor of size (batch_size, max_padding) containing the padded target sentences,
        where each sentence has special tokens <s> and </s> added and padding applied.

    Notes:
    - Each source and target sentence in the batch is first wrapped with a beginning-of-sentence (BOS_ID)
      and end-of-sentence (EOS_ID) token.
    - Padding is applied using the PAD_ID, and the resulting tensors are stacked together for
      batched processing.
    - This function assumes all input sentences are tokenized into integer token IDs.
    - torch.stack can be helpful.
    """

    # initialize special tokens  
    bs_id = torch.tensor([BOS_ID], device=device)
    eos_id = torch.tensor([EOS_ID], device=device)

    # initialize lists to hold processed data 
    src_list, tgt_list = [], []

    # TODO: Implement the collate_batch function.
    # YOUR CODE STARTS HERE
    def to_list(x):
        return x.tolist() if isinstance(x, torch.Tensor) else x

    def build_seq(ids):
        ids = to_list(ids)
        inner_max = max_padding - 2  # reserve slots for BOS/EOS
        trimmed = ids[:inner_max]
        seq = [BOS_ID] + trimmed + [EOS_ID]
        if len(seq) < max_padding:
            seq += [PAD_ID] * (max_padding - len(seq))
        return torch.tensor(seq, dtype=torch.long, device=device)

    src_tensor_list, tgt_tensor_list = [], []
    for src_ids, tgt_ids in batch_list:
        src_tensor_list.append(build_seq(src_ids))
        tgt_tensor_list.append(build_seq(tgt_ids))

    src = torch.stack(src_tensor_list, dim=0)  # [B, max_padding]
    tgt = torch.stack(tgt_tensor_list, dim=0)  # [B, max_padding]
    # YOUR CODE ENDS HERE

    # return batch 
    return Batch(src, tgt)


def create_dataloaders(dataset, device, batch_size, max_padding, shuffle=False):
    """ 
    Create dataloaders from a dataset. Apply padding and create masks. 

    Input Params: 
        - dataset (Dataset): dataset to create dataloader from. 
        - device (str): device to move tensors to. 
        - batch_size (int): batch size. 
        - max_padding (int): maximum padding length for sequences. 
        - shuffle (bool): whether to shuffle the data. 
    Returns: 
        - dataloader (DataLoader): dataloader for the dataset. 
    """
    # create dataloaders
    collate_fn = lambda batch: collate_batch(batch, device, max_padding)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


def init_dataloaders(training_hp, device):
    """
    Initialize dataloaders and vocabularies for training. 
    Input Params: 
        - training_hp (TrainingHyperParams): hyperparameters for training. 
        - device (str): device to move tensors to. 
    Returns: 
        - train_dataloader (DataLoader): dataloader for training data. 
        - valid_dataloader (DataLoader): dataloader for validation data. 
        - src_vocab (dict): vocabulary for source data. 
        - tgt_vocab (dict): vocabulary for target data. 
    """
    # load data from paths 
    train_data = read_pairs(training_hp.train_src_path, training_hp.train_tgt_path)
    valid_data = read_pairs(training_hp.valid_src_path, training_hp.valid_tgt_path)
    src_data = [sent[0] for sent in train_data] + [sent[0] for sent in valid_data]
    tgt_data = [sent[1] for sent in train_data] + [sent[1] for sent in valid_data]

    # build vocab 
    src_vocab = build_vocab(src_data, lang=SRC_LANG)
    tgt_vocab = build_vocab(tgt_data, lang=TGT_LANG)

    # create datasets 
    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab)
    valid_dataset = TranslationDataset(valid_data, src_vocab, tgt_vocab)

    # create dataloaders 
    train_dataloader = create_dataloaders(dataset=train_dataset,
                                          device=device,
                                          batch_size=training_hp.batch_size,
                                          max_padding=training_hp.max_padding,
                                          shuffle=True)
    valid_dataloader = create_dataloaders(dataset=valid_dataset,
                                          device=device,
                                          batch_size=training_hp.batch_size,
                                          max_padding=training_hp.max_padding,
                                          shuffle=False)

    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(valid_dataloader)}")
    return train_dataloader, valid_dataloader, src_vocab, tgt_vocab
