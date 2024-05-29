"""Classes for specifying probe pytorch modules.
Draws from https://github.com/john-hewitt/structural-probes"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertConfig, BertModel, BertForMaskedLM, PreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM
import logging
import sys, os

from utils import InitYAMLObject

logger = logging.getLogger(__name__)


class Probe(nn.Module, InitYAMLObject):

  def print_param_count(self):
    total_params = 0
    for param in self.parameters():
      total_params += np.prod(param.size())
    tqdm.write('Probe has {} parameters'.format(total_params))


################################################# Prompt Finetuning #################################################
class PromptFinetuneProbe(Probe, PreTrainedModel):
  """ PromptFinetune Bert-based model

  For a batch of sentences, computes all n scores
  for each sentence in the batch.
  Same structure as PromptSentenceClassifier in ntk/nlpmodels.py
  """
  yaml_tag = '!PromptFinetuneProbe'
  def __init__(self, args, freeze_layers, num_labels):
    # label_space_size = 2: number of classes
    print('Constructing PromptFinetuneProbe')
    torch.manual_seed(args['seed'])
    if 'roberta' in args['model_string']:
        self.config = RobertaConfig.from_pretrained(args['model_string'])
    else:
        self.config = BertConfig.from_pretrained(args['model_string'])
    # disable dropout so that there is no randomness
    self.config.hidden_dropout_prob = 0
    self.config.attention_probs_dropout_prob = 0
    super(PromptFinetuneProbe, self).__init__(self.config)
    self.args = args
    self.freeze_layers = freeze_layers
    self.num_labels = num_labels

    # freeze pooler and first few encoders
    if 'roberta' in args['model_string']:
        self.model = RobertaForMaskedLM.from_pretrained(args['model_string'], config=self.config)
        modules = [*self.model.roberta.encoder.layer[:self.freeze_layers]]
        if hasattr(self.model.roberta, 'pooler') and self.model.roberta.pooler is not None:
            for param in self.model.roberta.pooler.parameters():
                param.requires_grad = False
    else:
        self.model = BertForMaskedLM.from_pretrained(args['model_string'], config=self.config)
        modules = [*self.model.bert.encoder.layer[:self.freeze_layers]]
        if hasattr(self.model.bert, 'pooler') and self.model.bert.pooler is not None:
            for param in self.model.bert.pooler.parameters():
                param.requires_grad = False

    self.model_name = args['model_string']

    for module in modules:
      for param in module.parameters():
        param.requires_grad = False

    self.print_param_count()
    self.__name__ = 'PromptFinetuneProbe'
    self.model.to(self.args['device'])

  def init(self, label_word_list):
      """
      Only save the word list related embedding to save space complexity and time complexity
      Parameters
      ----------
      label_word_list: task-specific term
      -------
      """
      self.label_word_list = label_word_list

      new_bias = torch.zeros((self.num_labels))
      if 'roberta' in self.model_name:
          for i, index in enumerate(label_word_list):
              new_bias[i] = self.model.lm_head.bias.data[index]
          self.model.lm_head.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=True)
          self.model.lm_head.decoder.weight = nn.Parameter(
              self.model.roberta.embeddings.word_embeddings.weight[label_word_list, :])
          self.model.lm_head.decoder.bias.data.copy_(new_bias)
          self.model.lm_head.bias.requires_grad = False
      else:
          for i, index in enumerate(label_word_list):
              new_bias[i] = self.model.cls.predictions.bias.data[index]
          self.model.cls.predictions.decoder = nn.Linear(self.config.hidden_size, self.num_labels, bias=True)
          self.model.cls.predictions.decoder.weight = nn.Parameter(
              self.model.bert.embeddings.word_embeddings.weight[label_word_list, :])
          self.model.cls.predictions.decoder.bias.data.copy_(new_bias)
          self.model.cls.predictions.bias.requires_grad = False

      self.print_param_count()
      self.__name__ = 'PromptFinetuneProbe'
      self.model.to(self.args['device'])

  def forward(self, input_ids, attention_mask, token_type_ids, mask_pos):
    """ Computes all n label logits for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of logits of shape (batch_size, max_seq_len)
    """
    prediction_mask_scores = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
    prediction_mask_scores = prediction_mask_scores[torch.arange(prediction_mask_scores.size(0)), mask_pos.squeeze(-1)]
    return prediction_mask_scores


from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model
class PromptLLMProbe(Probe):
  """ PromptFinetune Llama-based model (Decoder-based)

  For a batch of sentences, computes all n scores
  for each sentence in the batch.
  Same structure as PromptLLM in ntk/nlpmodels.py
  """
  yaml_tag = '!PromptLLMProbe'
  def __init__(self, model_name, num_labels=2, seed=2023, device='cuda:0'):
    # model_name: meta-llama/Llama-2-7b-hf
    print('Constructing PromptLLMProbe')
    torch.manual_seed(seed)
    super(PromptLLMProbe, self).__init__()
    self.num_labels = num_labels
    self.config = LlamaConfig.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)
    # include two specialized tokens
    self.model.resize_token_embeddings(self.config.vocab_size + 2)
    self.device = device
    self.model_name = model_name
    self.__name__ = 'PromptLLMProbe'


  def init(self, label_word_list):
      """
      Only save the word list related embedding to save space complexity and time complexity
      Parameters
      ----------
      label_word_list: task-specific term
      -------
      """
      self.label_word_list = label_word_list

      self.model.lm_head = nn.Linear(self.config.hidden_size, self.num_labels, bias=False)
      self.model.lm_head.weight = nn.Parameter(self.model.model.embed_tokens.weight[label_word_list, :])
      self.peft_config = LoraConfig(
          r=16,
          lora_alpha=16,
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
          bias="none",
          modules_to_save=["lm_head"]
      )
      self.model = get_peft_model(self.model, self.peft_config)
      self.model.to(self.device)

  def forward(self, input_ids, attention_mask, mask_pos):
    """ Computes all n label logits for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of logits of shape (batch_size, max_seq_len)
    """
    logits = self.model(input_ids, attention_mask=attention_mask).logits
    logits = logits[torch.arange(logits.size(0)), mask_pos.squeeze(-1)]
    return logits


######################################## FreeShap (NTK Kernel Regression) ##############################################
from entks.ntk import compute_ntk, init_torch, process_args, slice_data
from entks.nlpmodels import SentenceClassifier, PromptSentenceClassifier, PromptLLM
from entks.ntk_regression import (NTKRegression, NTKRegression_correction_multiclass,
                                  fastNTKRegression, shapleyNTKRegression)
from easydict import EasyDict as edict
import pprint

class NTKProbe(Probe):
    """ eNTK for Transformer model

    For a batch of sentences, computes all n scores
    for each sentence in the batch.
    """
    yaml_tag = '!NTKProbe'

    def __init__(self, args, num_labels):
        print('Constructing NTKProbe')
        super(NTKProbe, self).__init__()

        args = edict(args)
        logging.info(f"args =\n{pprint.pformat(vars(args))}")

        self.args = args
        self.freeze_layers = args['num_frozen_layers']
        self.num_labels = num_labels

        model = args['model']
        if 'llama' in args['model']:
            self.model = PromptLLM(model_name=model, num_labels=self.num_labels, seed=args['seed'])
        else:
            if args['prompt']:
                self.model = PromptSentenceClassifier(model_name=model, num_frozen_layers=self.freeze_layers,
                                                      num_labels=self.num_labels, seed=args['seed'])
            else:
                self.model = SentenceClassifier(model_name=model, num_frozen_layers=self.freeze_layers,
                                                num_labels=self.num_labels, seed=args['seed'])

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_batches = (param_count - 1) // args.grad_chunksize + 1
        logging.info(f"Splitting {param_count} parameters into {param_batches} batches")

        self.print_param_count()
        self.__name__ = 'NTKProbe'
        self.model.to(args['device'])
        self.device = args['device']
        self.ntk = None
        self.train_labels = None
        self.debug = args['debug']
        self.correction = args['correction']
        self.single_kernel = args['single_kernel']
        self.approximate_ntk = None
        self.pre_inv = None
        self.args['signgd'] = False
        self.normalize = False

    def approximate(self, method='inv'):
        # method: diagonal, inv
        self.approximate_ntk = method

    def signgd(self):
        self.args['signgd'] = True

    def normalize_ntk(self):
        self.normalize = True

    def forward(self, ids, mask, token_type_ids):
        # Deep Learning model forward
        return self.model(ids, mask, token_type_ids).logits

    def compute_ntk(self, train_set, test_set):
        # compute the NTK matrix for train and test set
        self.model = self.model.to(self.device)
        kwargs = process_args(self.args)
        if self.debug:
            num_train = len(train_set)
            num_test = len(test_set)
            ntk = torch.zeros(num_train + num_test, num_train)
        else:
            if self.single_kernel:
                ntk = compute_ntk(self.model, train_set, test_set, **kwargs)
            else:
                ntk = compute_ntk(self.model, train_set, test_set, self.num_labels, **kwargs)
        self.ntk = ntk
        # normalize the ntk matrix due to numerical overflow
        if self.normalize:
            self.ntk = self.ntk / 10000
            print("current mean value: ", abs(self.ntk.mean()))
        self.train_labels = torch.tensor([i['label'] for i in train_set])
        if self.correction:
            self.get_correction(train_set, test_set)
        return ntk

    def get_cached_ntk(self, ntk):
        self.ntk = ntk
        print("current mean value: ", abs(self.ntk.mean()))
        # saved NTK are un-normalized; normalize to avoid numerical overflow
        if abs(self.ntk.mean()) > 1000:
            print("normalize while loading with mean value: ", abs(self.ntk.mean()))
            self.ntk = self.ntk / 10000

    def get_train_labels(self, train_set):
        self.train_labels = torch.tensor([i['label'] for i in train_set])

    def get_sliced_ntk(self, ntk, sampled_idx):
        # mainly for robustness experiment when need to slice NTK
        sampled_idx = np.array(sampled_idx)
        k_train = ntk[:, sampled_idx[:, None], sampled_idx]
        k_test = ntk[:, ntk.size(2):, :]
        k_test = k_test[:, :, sampled_idx]
        print(k_train.shape, k_test.shape)
        self.ntk = torch.tensor(np.concatenate((k_train, k_test), axis=1))
        self.train_labels = self.train_labels[sampled_idx]
        print("current mean value: ", abs(self.ntk.mean()))
        if abs(self.ntk.mean()) > 1000:
            print("normalize while loading with mean value: ", abs(self.ntk.mean()))
            self.ntk = self.ntk / 10000

    def get_correction(self, full_train_set, full_test_set):
        """
        Compute initial logits to serve as correction term
        Parameters
        ----------
        full_train_set: full train data
        -------
        """
        self.train_logits = torch.zeros(len(full_train_set), self.num_labels)
        loader = torch.utils.data.DataLoader(full_train_set, batch_size=self.args.loader_batch_size,
                                             num_workers=self.args.loader_num_workers,
                                             persistent_workers=False if self.args.loader_num_workers == 0 else None,
                                             shuffle=False)
        for i, data in enumerate(loader):
            with torch.no_grad():
                del data['label']
                data['input_ids'] = data['input_ids'].to(self.args.device)
                data['attention_mask'] = data['attention_mask'].to(self.args.device)
                data['token_type_ids'] = data['token_type_ids'].to(self.args.device)
                if 'mask_pos' in data:
                    data['mask_pos'] = data['mask_pos'].to(self.args.device)
                logits = self.model(data)
                self.train_logits[i * self.args.loader_batch_size: (i + 1) * self.args.loader_batch_size] = logits

        self.test_logits = torch.zeros(len(full_test_set), self.num_labels)
        loader = torch.utils.data.DataLoader(full_test_set, batch_size=self.args.loader_batch_size,
                                             num_workers=self.args.loader_num_workers,
                                             persistent_workers=False if self.args.loader_num_workers == 0 else None,
                                             shuffle=False)
        for i, data in enumerate(loader):
            with torch.no_grad():
                del data['label']
                data['input_ids'] = data['input_ids'].to(self.args.device)
                data['attention_mask'] = data['attention_mask'].to(self.args.device)
                data['token_type_ids'] = data['token_type_ids'].to(self.args.device)
                if 'mask_pos' in data:
                    data['mask_pos'] = data['mask_pos'].to(self.args.device)
                logits = self.model(data)
                self.test_logits[i * self.args.loader_batch_size: (i + 1) * self.args.loader_batch_size] = logits

    def kernel_regression(self, train_indices, test_set, per_point=False):
        # perform kernel regression for given train dataset and test dataset
        # select a proper submatrix from the full ntk matrix
        k_train = self.ntk[:, train_indices[:, None], train_indices]
        y_train = self.train_labels[train_indices]

        # construct the kernel matrix for the train set
        if self.correction:
            kr_model = NTKRegression_correction_multiclass(k_train, y_train, self.num_labels,
                                                           train_logits=self.train_logits[train_indices],
                                                           test_logits=self.test_logits)
        else:
            if self.approximate_ntk == 'diagonal':
                kr_model = fastNTKRegression(k_train, y_train, self.num_labels, batch_size=50)
            elif self.approximate_ntk == 'inv':
                if len(train_indices) % 500 == 0:
                    self.pre_inv = None
                kr_model = shapleyNTKRegression(k_train, y_train, self.num_labels, self.pre_inv)
            else:
                kr_model = NTKRegression(k_train, y_train, self.num_labels)

        k_test = self.ntk[:, self.ntk.size(2):, :]
        k_test = k_test[:, :, train_indices]
        # improvement note: 500 can be adjustable
        if self.approximate_ntk == 'inv' and len(train_indices) >= 500:
            test_preds, pre_inv = kr_model(k_test, return_inv=True)
            self.pre_inv = pre_inv
        else:
            test_preds = kr_model(k_test)
        test_preds = test_preds.to('cpu')
        test_labels = torch.tensor([i['label'] for i in test_set])

        if per_point:
            test_acc = (test_preds.argmax(dim=1) == test_labels).float()
            test_loss = F.cross_entropy(test_preds, test_labels, reduction='none').detach().cpu().numpy()
            return test_loss, test_acc
        # Compute accuracy on train and test sets
        test_acc = (test_preds.argmax(dim=1) == test_labels).float().mean()
        test_loss = F.cross_entropy(test_preds, test_labels, reduction='mean').item()
        # sanity check
        if test_loss > 1:
            print("bad kernel regression")
            print(
                f"train loss:, {F.cross_entropy(kr_model(k_train), y_train, reduction='mean').item()}, test loss: {test_loss}, test acc: {test_acc}")
        return test_loss, test_acc

    def kernel_regression_idx(self, train_indices, test_set, has_pre_inv=False):
        # perform kernel regression for given train dataset and test dataset, for robustness on large dataset
        #  select a proper submatrix from the full ntk matrix
        k_train = self.ntk[:, train_indices[:, None], train_indices]
        # print("train kernel shape:", k_train.shape)
        y_train = self.train_labels[train_indices]

        if not has_pre_inv:
            kr_model = shapleyNTKRegression(k_train, y_train, self.num_labels, None)
        else:
            kr_model = shapleyNTKRegression(k_train, y_train, self.num_labels, self.pre_inv)

        k_test = self.ntk[:, self.ntk.size(2):, :]
        k_test = k_test[:, :, train_indices]

        if not has_pre_inv:
            # print(len(train_indices))
            test_preds, pre_inv = kr_model(k_test, return_inv=True)
            self.pre_inv = pre_inv
        else:
            test_preds = kr_model(k_test)

        test_preds = test_preds.to('cpu')
        test_labels = torch.tensor([i['label'] for i in test_set])
        # Compute accuracy on train and test sets
        test_acc = (test_preds.argmax(dim=1) == test_labels).float().mean()

        test_loss = F.cross_entropy(test_preds, test_labels, reduction='mean').item()
        if test_loss > 1:
            print("bad kernel regression")
            print(
                f"train loss:, {F.cross_entropy(kr_model(k_train), y_train, reduction='mean').item()}, test loss: {test_loss}, test acc: {test_acc}")
        return test_loss, test_acc


############################################ Other baselines Helper class ##############################################
################################################# Standard Finetuning #################################################
class FinetuneProbe(Probe, PreTrainedModel):
  """
  Finetune Bert-based model
  """
  yaml_tag = '!FinetuneProbe'
  def __init__(self, args, freeze_layers, num_labels):
    # label_space_size = 2: number of classes
    print('Constructing FinetuneProbe')
    torch.manual_seed(args['seed'])
    config = BertConfig.from_pretrained(args['model_string'])
    # disable dropout so that there is no randomness
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    super(FinetuneProbe, self).__init__(config)
    self.args = args
    self.freeze_layers = freeze_layers
    self.num_labels = num_labels

    self.model = BertModel.from_pretrained(args['model_string'], config=config)
    # freeze the first num_frozen_layers layers of the model
    modules = [*self.model.encoder.layer[:self.freeze_layers]]
    for module in modules:
      for param in module.parameters():
        param.requires_grad = False
    # freeze the pooler
    if hasattr(self.model, 'pooler') and self.model.pooler is not None:
        for param in self.model.pooler.parameters():
            param.requires_grad = False

    # Dynamically find the size of the model's output (hidden state)
    hidden_size = self.model.config.hidden_size
    self.classifier = nn.Linear(hidden_size, self.num_labels)
    self.init_weights()
    self.print_param_count()
    self.__name__ = 'FinetuneProbe'
    self.model.to(args['device'])
    self.classifier.to(args['device'])

  def forward(self, ids, mask, token_type_ids):
      """ Computes all n label logits for each sentence in a batch.

      Args:
        batch: a batch of word representations of the shape
          (batch_size, max_seq_len, representation_dim)
      Returns:
        A tensor of logits of shape (batch_size, max_seq_len)
      """
      # Get sentence-level output ([CLS] representation)
      outputs = self.model(ids, mask, token_type_ids)
      cls_output = outputs.last_hidden_state[:, 0, :]
      # Pass through classifier
      logits = self.classifier(cls_output)
      return logits