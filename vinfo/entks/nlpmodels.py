from transformers import BertForSequenceClassification, BertConfig, BertModel, BertForMaskedLM, PreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM
import torch
from torch import nn

################################################# Standard Finetuning #################################################
class SentenceClassifier(PreTrainedModel):
    def __init__(self, model_name, num_frozen_layers=8, num_labels=2, seed=2023):
        torch.manual_seed(seed)
        config = BertConfig.from_pretrained(model_name)
        # disable dropout so that there is no randomness
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        super(SentenceClassifier, self).__init__(config)

        # Load pretrained model and tokenizer
        self.model = BertModel.from_pretrained(model_name, config=config)
        # freeze the first num_frozen_layers layers of the model
        for i in range(num_frozen_layers):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = False
        # freeze the pooler
        if hasattr(self.model, 'pooler') and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = False

        # Dynamically find the size of the model's output (hidden state)
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.init_weights()

    def forward(self, inputs):

        # Get sentence-level output ([CLS] representation)
        outputs = self.model(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Pass through classifier
        logits = self.classifier(cls_output)
        return logits

################################################# Prompt Finetuning #################################################
class PromptSentenceClassifier(PreTrainedModel):
  """ PromptFinetune Bert-based model

  For a batch of sentences, computes all n scores
  for each sentence in the batch.
  Same structure as PromptFinetuneProbe in probe.py
  """
  yaml_tag = '!PromptSentenceClassifier'
  def __init__(self, model_name, num_frozen_layers=8, num_labels=2, seed=2023):
    # label_space_size = 2: number of classes
    print('Constructing PromptFinetuneProbe')
    torch.manual_seed(seed)
    if 'roberta' in model_name:
        self.config = RobertaConfig.from_pretrained(model_name)
    else:
        self.config = BertConfig.from_pretrained(model_name)
    # disable dropout so that there is no randomness
    self.config.hidden_dropout_prob = 0
    self.config.attention_probs_dropout_prob = 0
    super(PromptSentenceClassifier, self).__init__(self.config)
    self.freeze_layers = num_frozen_layers
    self.num_labels = num_labels

    # freeze pooler and first few encoders
    if 'roberta' in model_name:
        self.model = RobertaForMaskedLM.from_pretrained(model_name, config=self.config)
        modules = [*self.model.roberta.encoder.layer[:self.freeze_layers]]
        if hasattr(self.model.roberta, 'pooler') and self.model.roberta.pooler is not None:
            for param in self.model.roberta.pooler.parameters():
                param.requires_grad = False
    else:
        self.model = BertForMaskedLM.from_pretrained(model_name, config=self.config)
        modules = [*self.model.bert.encoder.layer[:self.freeze_layers]]
        if hasattr(self.model.bert, 'pooler') and self.model.bert.pooler is not None:
            for param in self.model.bert.pooler.parameters():
                param.requires_grad = False

    for module in modules:
      for param in module.parameters():
        param.requires_grad = False
            
    self.model_name = model_name

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


  def forward(self, inputs):
    """ Computes all n label logits for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of logits of shape (batch_size, max_seq_len)
    """
    logits = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids']).logits
    logits = logits[torch.arange(logits.size(0)), inputs['mask_pos'].squeeze(-1)]
    return logits


from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model
class PromptLLM(nn.Module):
  """ PromptFinetune Llama-based model (Decoder-based)

  For a batch of sentences, computes all n scores
  for each sentence in the batch.
  Same structure as PromptLLMProbe in probe.py
  """
  yaml_tag = '!PromptLLM'
  def __init__(self, model_name, num_labels=2, seed=2023):
    # model_name: meta-llama/Llama-2-7b-hf
    print('Constructing PromptLLM')
    torch.manual_seed(seed)
    super(PromptLLM, self).__init__()
    self.num_labels = num_labels
    self.config = LlamaConfig.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)
    self.model.resize_token_embeddings(self.config.vocab_size + 2)


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

  def forward(self, inputs):
    """ Computes all n label logits for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of logits of shape (batch_size, max_seq_len)
    """
    logits = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
    logits = logits[torch.arange(logits.size(0)), inputs['mask_pos'].squeeze(-1)]
    return logits