import os
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import Levenshtein as levenshtein

from tqdm import tqdm
from yaml import YAMLObject
from transformers import AutoTokenizer, AutoModel
from torchvision.datasets import MNIST
import pandas as pd
from utils import TRAIN_STR, DEV_STR, TEST_STR, IID_STR, OOD_STR, InitYAMLObject
import numpy as np
import logging

"""
Classes for loading, caching, and yielding text datasets
"""

class TransformerDataset(Dataset):
  """
  Standard Finetuning Dataset, return tokenized texts
  """
  def __init__(self, dataset, tokenizer, max_length=128):
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.dataset = list(dataset)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    sentences, label = self.dataset[idx]
    if len(sentences) == 2:
        sentence1, sentence2 = sentences
        inputs = self.tokenizer.encode_plus(sentence1, sentence2,
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            add_special_tokens=True,
                                            return_token_type_ids=True)
    else:
        sentence = sentences[0]
        inputs = self.tokenizer.encode_plus(sentence, None,
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            add_special_tokens=True,
                                            return_token_type_ids=True)


    return {'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
            }


class PropmtDataset(Dataset):
  """
    Prompt Finetuning Dataset, return tokenized texts in form: [text+prompts]
    make use of codes from https://github.com/princeton-nlp/LM-Kernel-FT
  """
  def __init__(self, dataset, tokenizer, template, label_word_list, max_length=128, first_sent_limit=None,
               other_sent_limit=None, truncate_head=False, support_labels=None):
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.dataset = list(dataset)
    self.label_word_list = label_word_list
    self.template = template
    self.first_sent_limit = first_sent_limit
    self.other_sent_limit = other_sent_limit
    self.truncate_head = truncate_head
    self.support_labels = support_labels

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):

    def enc(text):
      return self.tokenizer.encode(text, add_special_tokens=False)
    input_text_list, label = self.dataset[idx]
    if 'uncased' in type(self.tokenizer).__name__.lower():
      input_text_list = [x.lower() for x in input_text_list]
    if 'llama' in type(self.tokenizer).__name__.lower():
      special_tokens_dict = {"mask_token": "<MASK>", "sep_token": "<SEP>"}
      self.tokenizer.add_special_tokens(special_tokens_dict)

    input_ids = []
    attention_mask = []
    token_type_ids = []  # For BERT and RoBERTa
    """
    Concatenate all sentences and prompts based on the provided template.
    Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
    Template example: ''*cls**sent_0*_It_was*mask*.*sep+*'' (Bert for sst-2)
    *xx* represent variables:
        *cls*: cls_token
        *mask*: mask_token
        *sep*: sep_token
        *sep+*: sep_token, also means +1 for segment id
        *sent_i*: sentence i (input_text_list[i])
        *sent-_i*: same as above, but delete the last token
        *sentl_i*: same as above, but use lower case for the first word
        *sentl-_i*: same as above, but use lower case for the first word and delete the last token
        *+sent_i*: same as above, but add a space before the sentence
        *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
        *label_i*: label_word_list[i]
        *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

    Use "_" to replace space.
    PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
    """
    assert self.template is not None

    special_token_mapping = {
      'cls': self.tokenizer.cls_token_id, 'mask': self.tokenizer.mask_token_id, 'sep': self.tokenizer.sep_token_id,
      'sep+': self.tokenizer.sep_token_id, '<s>': self.tokenizer.bos_token_id
    }
    template_list = self.template.split('*')  # Get variable list in the self.template
    segment_id = 0  # Current segment id. Segment id +1 if encountering sep+.
    for part_id, part in enumerate(template_list):
      new_tokens = []
      segment_plus_1_flag = False
      if part in special_token_mapping:
        if part == 'cls' and 'T5' in type(self.tokenizer).__name__:
          # T5 does not have cls token
          continue
        new_tokens.append(special_token_mapping[part])
        if part == 'sep+':
          segment_plus_1_flag = True
      elif part[:6] == 'label_':
        # Note that label_word_list already has extra space, so do not add more space ahead of it.
        label_id = int(part.split('_')[1])
        label_word = self.label_word_list[label_id]
        new_tokens.append(label_word)
      elif part[:7] == 'labelx_':
        instance_id = int(part.split('_')[1])
        label_id = self.support_labels[instance_id]
        label_word = self.label_word_list[label_id]
        new_tokens.append(label_word)
      elif part[:5] == 'sent_':
        sent_id = int(part.split('_')[1])
        new_tokens += enc(input_text_list[sent_id])
      elif part[:6] == '+sent_':
        # Add space
        sent_id = int(part.split('_')[1])
        new_tokens += enc(' ' + input_text_list[sent_id])
      elif part[:6] == 'sent-_':
        # Delete the last token
        sent_id = int(part.split('_')[1])
        new_tokens += enc(input_text_list[sent_id][:-1])
      elif part[:6] == 'sentl_':
        # Lower case the first token
        sent_id = int(part.split('_')[1])
        text = input_text_list[sent_id]
        text = text[:1].lower() + text[1:]
        new_tokens += enc(text)
      elif part[:7] == '+sentl_':
        # Lower case the first token and add space
        sent_id = int(part.split('_')[1])
        text = input_text_list[sent_id]
        text = text[:1].lower() + text[1:]
        new_tokens += enc(' ' + text)
      elif part[:7] == 'sentl-_':
        # Lower case the first token and discard the last token
        sent_id = int(part.split('_')[1])
        text = input_text_list[sent_id]
        text = text[:1].lower() + text[1:]
        new_tokens += enc(text[:-1])
      elif part[:6] == 'sentu_':
        # Upper case the first token
        sent_id = int(part.split('_')[1])
        text = input_text_list[sent_id]
        text = text[:1].upper() + text[1:]
        new_tokens += enc(text)
      elif part[:7] == '+sentu_':
        # Upper case the first token and add space
        sent_id = int(part.split('_')[1])
        text = input_text_list[sent_id]
        text = text[:1].upper() + text[1:]
        new_tokens += enc(' ' + text)
      else:
        # Just natural language prompt
        part = part.replace('_', ' ')
        # handle special case when T5 tokenizer might add an extra space
        if len(part) == 1:
          new_tokens.append(self.tokenizer.convert_tokens_to_ids(part))
        else:
          new_tokens += enc(part)

      if part[:4] == 'sent' or part[1:5] == 'sent':
        # If this part is the sentence, limit the sentence length
        sent_id = int(part.split('_')[1])
        if sent_id == 0:
          if self.first_sent_limit is not None:
            new_tokens = new_tokens[:self.first_sent_limit]
        else:
          if self.other_sent_limit is not None:
            new_tokens = new_tokens[:self.other_sent_limit]

      input_ids += new_tokens
      attention_mask += [1 for _ in range(len(new_tokens))]
      token_type_ids += [segment_id for _ in range(len(new_tokens))]

      if segment_plus_1_flag:
        segment_id += 1
    # Padding
    if self.first_sent_limit is not None and len(input_ids) > self.max_length:
      # If using sentence limit, the total length still exceeds the maximum limit, report a warning
      # print("Input exceeds max_length limit: {}".format(self.tokenizer.decode(input_ids)))
      pass

    # Find mask token
    # Make sure that the masked position is inside the max_length
    assert self.tokenizer.mask_token_id in input_ids, \
      "Mask token not found for input: {} {}".format(input_text_list, input_ids)
    mask_pos = [input_ids.index(self.tokenizer.mask_token_id)]

    # Truncate
    if len(input_ids) > self.max_length:
      if self.truncate_head:
        input_ids = input_ids[-self.max_length:]
        attention_mask = attention_mask[-self.max_length:]
        token_type_ids = token_type_ids[-self.max_length:]
      else:
        # Default is to truncate the tail
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        token_type_ids = token_type_ids[:self.max_length]

    assert mask_pos[0] < self.max_length, \
      "Mask token {} exceed max length {}".format(mask_pos[0], self.max_length)

    result = {'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mask_pos': torch.tensor(mask_pos, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
            }

    # padding
    pad_len = self.max_length - len(input_ids)
    result['input_ids'] = F.pad(result['input_ids'], (0, pad_len), 'constant', 0)
    result['attention_mask'] = F.pad(result['attention_mask'], (0, pad_len), 'constant', 0)
    if 'bert' in type(self.tokenizer).__name__.lower():
      # Provide token type ids for BERT and Roberta
      result['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
      result['token_type_ids'] = F.pad(result['token_type_ids'], (0, pad_len), 'constant', 0)
    return result


class ListDataset(Dataset, InitYAMLObject):
  """
  Container class for non-NTK fine-tuning
  """
  yaml_tag = '!ListDataset'
  def __init__(self, args, data_loader, prompt=False, mapping=None,
               template=None, first_sent_limit=None, other_sent_limit=None):
    """
    Arguments:
      output_datset: 
    """
    self.args = args
    self.data_loader = data_loader
    self.train_data = None
    self.dev_data = None
    self.test_data = None
    self.tokenizer = AutoTokenizer.from_pretrained(args['model_string'])
    self.prompt = prompt
    self.dev_str = DEV_STR
    if self.data_loader.dataset_name == "mnli":
      self.dev_str = IID_STR
    elif self.data_loader.dataset_name == "subj" or self.data_loader.dataset_name == "ag_news":
      self.dev_str = TEST_STR
    if self.prompt:
      self.label_to_word = eval(mapping)
      for key in self.label_to_word:
        # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
        if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
          # Make sure space+word is in the vocabulary
          assert len(self.tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
          self.label_to_word[key] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(' ' + self.label_to_word[key])[0])
        else:
          self.label_to_word[key] = self.label_to_word[key].strip('<[.,')
          assert len(self.tokenizer.tokenize(self.label_to_word[key])) == 1
          self.label_to_word[key] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.label_to_word[key])[0])
        print(f"Label {key} to word {self.tokenizer.convert_ids_to_tokens(self.label_to_word[key])} ({self.label_to_word[key]})")
      self.label_word_list = list(self.label_to_word.values())
      print("label_to_word: ", self.label_to_word)
      print("label_list: ", self.label_word_list)
      self.template = template
      self.first_sent_limit = first_sent_limit
      self.other_sent_limit = other_sent_limit

  def _load_data(self, split_string):
    """
    Loads data from disk into RAM tensors for passing to a network on GPU
    Tokenize the data, then convert to tensors, then cache the tensors
    """
    # yield_dataset: [[data['sentence'], data['label']]]
    # truncation, max_length, and padding are used to make sure all sequences have the same length
    for *sentence, label in tqdm(self.data_loader.yield_dataset(split_string), desc='[loading]'):
      yield sentence, label


  def get_idx_dataset(self, idxs, split="train"):
    if split == "train":
      if self.train_data is None:
        self.train_data = list(self._load_data(TRAIN_STR))
      if self.prompt:
        generator = PropmtDataset(self.train_data, self.tokenizer, self.template, self.label_word_list,
                                  max_length=self.args['seq_len'], first_sent_limit=self.first_sent_limit,
                                  other_sent_limit=self.other_sent_limit)
      else:
        generator = TransformerDataset(self.train_data, self.tokenizer, max_length=self.args['seq_len'])
      self.train_generator = Subset(generator, idxs)
      return self.train_generator
    elif split == "val":
      if self.dev_data is None:
        self.dev_data = list(self._load_data(self.dev_str))
      if self.prompt:
        generator = PropmtDataset(self.dev_data, self.tokenizer, self.template, self.label_word_list,
                                  max_length=self.args['seq_len'], first_sent_limit=self.first_sent_limit,
                                  other_sent_limit=self.other_sent_limit)
      else:
        generator = TransformerDataset(self.dev_data, self.tokenizer, max_length=self.args['seq_len'])
      self.val_generator = Subset(generator, idxs)
      return self.val_generator

  def get_idx_dataloader(self, idxs, split="train"):
    """
    Firstly tokenize the sentence, then get dataloader: return a DataLoader
    train_data/dev_data: whole dataset
    train_generator/dev_generator: sliced data
    """

    if split == "train":
      if self.train_data is None:
        self.train_data = list(self._load_data(TRAIN_STR))
      if self.prompt:
        generator = PropmtDataset(self.train_data, self.tokenizer, self.template, self.label_word_list,
                                  max_length=self.args['seq_len'], first_sent_limit=self.first_sent_limit,
                                  other_sent_limit=self.other_sent_limit)
      else:
        generator = TransformerDataset(self.train_data, self.tokenizer, max_length=self.args['seq_len'])
      batch_size = self.args['batchsize']
      self.train_generator = Subset(generator, idxs)
    elif split == "val":
      if self.dev_data is None:
        self.dev_data = list(self._load_data(self.dev_str))
      if self.prompt:
        generator = PropmtDataset(self.dev_data, self.tokenizer, self.template, self.label_word_list,
                                  max_length=self.args['seq_len'], first_sent_limit=self.first_sent_limit,
                                  other_sent_limit=self.other_sent_limit)
      else:
        generator = TransformerDataset(self.dev_data, self.tokenizer, max_length=self.args['seq_len'])
      batch_size = 3 * self.args['batchsize']
      self.val_generator = Subset(generator, idxs)
    subset = Subset(generator, idxs)
    del generator
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


  def get_idx_dataloader_reindx(self, idxs, split="train"):
    """
    Retrieve subsets of the data by index
    """
    if split == "train":
      generator = self.train_generator
      batch_size = self.args['batchsize']
    elif split == "val":
      generator = self.val_generator
      batch_size = 3 * self.args['batchsize']
    subset = Subset(generator, idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


class Loader(InitYAMLObject):
  """
  Base class for objects that read datasets from disk
  and yield sentence buffers for tokenization and labeling
  Strictly for description
  """
  yaml_tag = '!Loader'


class EasyReader(Loader):
  yaml_tag = '!EasyReader'
  def __init__(self, args):
    self.dataset_name = args['dataset_name']
    if 'data_poison' in args:
      self.data_poison = args['data_poison']
    else:
      self.data_poison = False

  def yield_dataset(self, split_string):
    from datasets import load_dataset
    """
    Yield a dataset
    """
    data_poison = self.data_poison
    if split_string != "train":
      data_poison = False
    if self.dataset_name == "sst2":
      dataset = load_dataset("sst2", split=split_string)
    elif self.dataset_name == "mr":
      dataset = load_dataset("rotten_tomatoes", split=split_string)
    elif self.dataset_name == "subj":
      dataset = load_dataset("SetFit/subj", split=split_string)
    elif self.dataset_name == "ag_news":
      dataset = load_dataset("ag_news", split=split_string)
    elif self.dataset_name == "rte":
      dataset = load_dataset("glue", "rte", split=split_string)
    elif self.dataset_name == "mnli":
      dataset = load_dataset("glue", "mnli", split=split_string)
    elif self.dataset_name == "hans":
      dataset = load_dataset("hans", split=split_string)
    elif self.dataset_name == "mrpc":
      dataset = load_dataset("glue", "mrpc", split=split_string)

    if split_string == "train":
      dataset = dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)

    nrows = dataset.num_rows
    print(f"{self.dataset_name}: {nrows}")
    if data_poison:
      num_to_flip = int(nrows * 0.1)
      import random
      random.seed(2023)
      indices_to_flip = random.sample(range(nrows), num_to_flip)

    if self.dataset_name == "sst2":
      for data in dataset:
        if data_poison and data['idx'] in indices_to_flip:
          yield data['sentence'], 1 - data['label']
        else:
          yield data['sentence'], data['label']
    elif self.dataset_name == "mr":
      for data in dataset:
        if data_poison and data['idx'] in indices_to_flip:
          yield data['text'], 1 - data['label']
        else:
          yield data['text'], data['label']
    elif self.dataset_name == "subj":
      for data in dataset:
          if data_poison and data['idx'] in indices_to_flip:
            yield data['text'], 1 - data['label']
          else:
            yield data['text'], data['label']
    elif self.dataset_name == "ag_news":
      for data in dataset:
          if data_poison and data['idx'] in indices_to_flip:
            yield data['text'],  (1+data['label'])%4
          else:
            yield data['text'], data['label']
    elif self.dataset_name == "rte":
      for data in dataset:
        if data_poison and data['idx'] in indices_to_flip:
          yield data['sentence1'], data['sentence2'], 1 - data['label']
        else:
          yield data['sentence1'], data['sentence2'], data['label']
    elif self.dataset_name == "mnli":
      for data in dataset:
        if data_poison and data['idx'] in indices_to_flip:
          if data['label'] != 2:
            corrupted_label = 2
          else:
            corrupted_label = 0
          yield data['premise'], data['hypothesis'], corrupted_label
        else:
          yield data['premise'], data['hypothesis'], data['label']
    elif self.dataset_name == "hans":
      for data in dataset:
        if data_poison and data['idx'] in indices_to_flip:
          if data['label'] != 2:
            corrupted_label = 2
          else:
            corrupted_label = 0
          yield data['premise'], data['hypothesis'], corrupted_label
        else:
          yield data['premise'], data['hypothesis'], data['label']
    elif self.dataset_name == "mrpc":
      for data in dataset:
        if data_poison and data['idx'] in indices_to_flip:
          yield data['sentence1'], data['sentence2'], 1 - data['label']
        else:
          yield data['sentence1'], data['sentence2'], data['label']


class FastListDataset(Dataset, InitYAMLObject):
  """
  Container class for NTK fine-tuning
  """
  yaml_tag = '!FastListDataset'
  def __init__(self, args, data_loader, prompt=False, mapping=None, template=None, first_sent_limit=None,
               other_sent_limit=None):
    """
    Arguments:
      output_datset: 
    """
    self.args = args
    self.data_loader = data_loader
    self.train_data = None
    self.dev_data = None
    self.test_data = None
    self.tokenizer = AutoTokenizer.from_pretrained(args['model_string'])
    self.prompt = prompt
    self.dev_str = DEV_STR
    if self.data_loader.dataset_name == "mnli":
      self.dev_str = IID_STR
    elif self.data_loader.dataset_name == "subj" or self.data_loader.dataset_name == "ag_news":
      self.dev_str = TEST_STR
    if self.prompt:
      self.label_to_word = eval(mapping)
      for key in self.label_to_word:
        # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
        if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
          # Make sure space+word is in the vocabulary
          assert len(self.tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
          self.label_to_word[key] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(' ' + self.label_to_word[key])[0])
        else:
          self.label_to_word[key] = self.label_to_word[key].strip('<[.,')
          assert len(self.tokenizer.tokenize(self.label_to_word[key])) == 1
          self.label_to_word[key] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.label_to_word[key])[0])
          # self.label_to_word[key] = self.tokenizer.convert_tokens_to_ids(self.label_to_word[key])
        print(f"Label {key} to word {self.tokenizer.convert_ids_to_tokens(self.label_to_word[key])} ({self.label_to_word[key]})")
      self.label_word_list = list(self.label_to_word.values())
      print("label_to_word: ", self.label_to_word)
      print("label_list: ", self.label_word_list)
      self.template = template
      self.first_sent_limit = first_sent_limit
      self.other_sent_limit = other_sent_limit

  def _load_data(self, split_string):
    """
    Loads data from disk into RAM tensors for passing to a network on GPU
    Tokenize the data, then convert to tensors, then cache the tensors
    """
    # yield_dataset: [[data['sentence'], data['label']]]
    # truncation, max_length, and padding are used to make sure all sequences have the same length
    for *sentence, label in tqdm(self.data_loader.yield_dataset(split_string), desc='[loading]'):
      yield sentence, label

  def get_idx_dataset(self, idx, split="train"):
    """
    Get a dataset by index
    """
    if split == "train":
      if self.train_data is None:
        data = list(self._load_data(TRAIN_STR))
        if self.prompt:
          self.train_data = PropmtDataset(data, self.tokenizer, self.template, self.label_word_list,
                                          max_length=self.args['seq_len'], first_sent_limit=self.first_sent_limit,
                                          other_sent_limit=self.other_sent_limit)
        else:
          self.train_data = TransformerDataset(data, self.tokenizer, max_length=self.args['seq_len'])
      return SubsetDataset(self.train_data, idx)
    elif split == "val":
      if self.dev_data is None:
        data = list(self._load_data(self.dev_str))
        if self.prompt:
          self.dev_data = PropmtDataset(data, self.tokenizer, self.template, self.label_word_list,
                                        max_length=self.args['seq_len'], first_sent_limit=self.first_sent_limit,
                                        other_sent_limit=self.other_sent_limit)
        else:
          self.dev_data = TransformerDataset(data, self.tokenizer, max_length=self.args['seq_len'])
      return SubsetDataset(self.dev_data, idx)

  def get_idx_dataset_large(self, idx, split="train"):
    """
    Able to process for large datasets, over 20k
    previous method can lead of OOM
    this process can lead to slower kernel regression
    """
    if split == "train":
      if self.train_data is None:
        data = list(self._load_data(TRAIN_STR))
        data = [data[i] for i in idx] # for large data processing; cannot align with robustness code
        if self.prompt:
          self.train_data = PropmtDataset(data, self.tokenizer, self.template, self.label_word_list,
                                          max_length=self.args['seq_len'], first_sent_limit=self.first_sent_limit,
                                          other_sent_limit=self.other_sent_limit)
        else:
          self.train_data = TransformerDataset(data, self.tokenizer, max_length=self.args['seq_len'])
      return self.train_data
    elif split == "val":
      if self.dev_data is None:
        data = list(self._load_data(self.dev_str))
        data = [data[i] for i in idx]
        if self.prompt:
          self.dev_data = PropmtDataset(data, self.tokenizer, self.template, self.label_word_list,
                                        max_length=self.args['seq_len'], first_sent_limit=self.first_sent_limit,
                                        other_sent_limit=self.other_sent_limit)
        else:
          self.dev_data = TransformerDataset(data, self.tokenizer, max_length=self.args['seq_len'])
      return self.dev_data


class SubsetDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.data = [original_dataset[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]