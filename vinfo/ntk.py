import pickle
import yaml as yaml
import click
from datetime import datetime
import random
from datasets import load_dataset, concatenate_datasets, Value
import torch

import sys

sys.path.insert(0, './lmntk')
sys.path.insert(0, './vinfo/lmntk')
# sys.path.insert(0, './vinfo/entks')

ontonotes_fields = ["one_offset_word_index", "token", "None", "ptb_pos", "ptb_pos2", "None2", "dep_rel", "None3",
                    "None4", "source_file", "part_number", "zero_offset_word_index", "token2", "ptb_pos3", "parse_bit",
                    "predicate_lemma", "predicate_frameset_id", "word_sense", "speaker_author", "named_entities"]

from dataset import *
from probe import *
from dvutils.Data_Shapley import *
import logging

from dataclasses import dataclass, field
from transformers import HfArgumentParser

import time


@dataclass
class DshapArguments:
    yaml_path: str = field(default='configs/dshap/$DATASET/$FILENAME.yaml', metadata={"help": "Path to YAML file."})
    just_cache_data: int = field(default=0, metadata={"help": "If 1, just writes data to cache; does not run experiment."})
    dataset_name: str = field(default='sst2', metadata={"help": "Dataset name."})
    num_dp: int = field(default=1000, metadata={"help": "Number of data points."})
    val_sample_num: int = field(default=900000, metadata={"help": "Number of validation data points."})
    tmc_iter: int = field(default=100, metadata={"help": "Number of iterations."})
    seed: int = field(default=2023, metadata={"help": "Random seed."})
    tmc_seed: int = field(default=2023, metadata={"help": "Random seed for tmc"})
    file_path: str = field(default='../robust/ntk/', metadata={"help": "Place to store the results."})
    prompt: bool = field(default=True, metadata={"help": "Whether to use prompt."})
    approximate: str = field(default="none", metadata={"help": "Whether to approximate."})
    run_shapley: bool = field(default=True, metadata={"help": "Whether to run shapley value."})
    per_point: bool = field(default=False, metadata={"help": "Whether to run shapley value per point."})
    posion: bool = field(default=False, metadata={"help": "Whether to introduce data posioning."})
    signgd: bool = field(default=False, metadata={"help": "Whether to use signgd kernel."})
    early_stopping: bool = field(default=False, metadata={"help": "Whether to early stop."})

def run_yaml_experiment(yaml_path, just_cache_data, dataset_name, num_dp, val_sample_num, tmc_iter, seed, tmc_seed, file_path,
                        prompt, approximate, run_shapley, per_point, posion, signgd, early_stopping):
    """
    Runs an experiment as configured by a yaml config file
    """
    import os
    os.environ["OMP_NUM_THREADS"] = '16'
    os.environ["OPENBLAS_NUM_THREADS"] = '16'
    os.environ["MKL_NUM_THREADS"] = '16'

    # Check if the folder exists, if not, create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"Folder '{file_path}' created.")
    else:
        print(f"Folder '{file_path}' already exists.")

    # Set global torch seed for model initialization etc.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Take constructed classes from yaml
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args['dataset']
    probe_model = yaml_args['probe_com']
    dshap_com = yaml_args['dshap_com']
    if prompt:
        probe_model.model.init(list_dataset.label_word_list)
    if approximate != "none":
        probe_model.approximate(approximate)
        if approximate == "inv":
            probe_model.normalize_ntk()
    if signgd:
        probe_model.signgd()

    if just_cache_data:
        print("Data caching done. Exiting...")
        return

    # sample a set of data points to conduct data valuation
    np.random.seed(seed)
    random.seed(seed)
    if dataset_name == "sst2":
        dataset = load_dataset("sst2")
    elif dataset_name == "mr":
        dataset = load_dataset("rotten_tomatoes")
    elif dataset_name == "subj":
        dataset = load_dataset("SetFit/subj")
    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
    elif dataset_name == "rte":
        dataset = load_dataset("glue", "rte")
    elif dataset_name == "mnli":
        dataset = load_dataset("glue", "mnli")
    elif dataset_name == "hans":
        dataset = load_dataset("hans")
    elif dataset_name == "mrpc":
        dataset = load_dataset("glue", "mrpc")
    # Sample 10 data points from the dataset
    train_data = dataset['train']
    train_data = train_data.map(lambda example, idx: {'idx': idx}, with_indices=True)
    train_data = train_data.shuffle(seed).select(range(min(train_data.num_rows, num_dp)))
    sampled_idx = train_data['idx']
    if dataset_name == "mnli":
        val_num = dataset['validation_matched'].num_rows
    elif dataset_name == "subj" or dataset_name == "ag_news":
        val_num = dataset['test'].num_rows
    else:
        val_num = dataset['validation'].num_rows
    if val_sample_num > val_num:
        sampled_val_idx = np.arange(val_num)
    else:
        sampled_val_idx = np.random.choice(np.arange(val_num), val_sample_num, replace=False).tolist()

    if 'llama' in probe_model.args['model']:
        model_name = 'llama'
    elif 'roberta' in probe_model.args['model']:
        model_name = 'roberta'
    elif 'bert' in probe_model.args['model']:
        model_name = 'bert'
    # data Shapley with entk
    print(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl")
    try:
        with open(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl", "rb") as f:
            ntk = pickle.load(f)
        print("++++++++++++++++++++++++++++++++++++using cached ntk++++++++++++++++++++++++++++++++++++")
        probe_model.get_cached_ntk(ntk)
        probe_model.get_train_labels(list_dataset.get_idx_dataset(sampled_idx, split="train"))
    except:
        print("++++++++++++++++++++++++++++++++++no cached ntk, computing+++++++++++++++++++++++++++++++++++")
        train_set = list_dataset.get_idx_dataset(sampled_idx, split="train")
        val_set = list_dataset.get_idx_dataset(sampled_val_idx, split="val")
        # Given that train_loader and val_loader are provided in run(), prepare datasets
        # Set parameters for ntk computation
        # compute ntk matrix
        ntk = probe_model.compute_ntk(train_set, val_set)
        # save the ntk matrix
        with open(f"{file_path}{dataset_name}_{model_name}_ntk_seed{seed}_num{num_dp}_sign{signgd}.pkl", "wb") as f:
            pickle.dump(ntk, f)
        print("++++++++++++++++++++++++++++++++++saving ntk cache+++++++++++++++++++++++++++++++++++")
    if run_shapley:
        checkpoint=True
        if per_point:
            checkpoint=False
        print("early_stopping", early_stopping)
        dv_result = dshap_com.run(data_idx=sampled_idx, val_data_idx=sampled_val_idx, iteration=tmc_iter,
                                  use_cache_ntk=True, prompt=prompt, seed=tmc_seed, num_dp=num_dp,
                                  checkpoint=checkpoint, per_point=per_point, early_stopping=early_stopping)

        mc_com = np.array(dshap_com.mc_cache)
        if per_point:
            result_dict = {'dv_result': dv_result,  # entropy, accuracy
                           'sampled_idx': sampled_idx}
            with open(f"{file_path}{dataset_name}_{model_name}_shapley_result_seed{seed}_num{num_dp}_appro{approximate}_sign{signgd}_earlystop{early_stopping}_tmc{tmc_seed}_iter{tmc_iter}.pkl",
                      "wb") as f:
                pickle.dump(result_dict, f)
        else:
            ac_com = np.array(dshap_com.ac_cache)
            result_dict = {'dv_result': dv_result,  # entropy, accuracy
                           'mc_com': mc_com,
                           'ac_com': ac_com,
                           'sampled_idx': sampled_idx}
            with open(f"{file_path}{dataset_name}_{model_name}_shapley_result_seed{seed}_num{num_dp}_appro{approximate}_sign{signgd}_posion{posion}_earlystop{early_stopping}_tmc{tmc_seed}_iter{tmc_iter}.pkl", "wb") as f:
                pickle.dump(result_dict, f)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    from torch.multiprocessing import set_start_method, set_sharing_strategy
    import torch.multiprocessing as mp

    set_start_method("spawn")
    set_sharing_strategy("file_system")

    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # run_yaml_experiment()

    parser = HfArgumentParser(DshapArguments)
    args, remaining_argv = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    sys.argv = [sys.argv[0]] + remaining_argv
    # run_yaml_experiment(args.yaml_path, args.just_cache_data, args.num_dp, args.val_sample_num, args.tmc_iter, args.seed, args.file_path)
    run_yaml_experiment(args.yaml_path, args.just_cache_data, args.dataset_name, args.num_dp, args.val_sample_num,
                        args.tmc_iter, args.seed, args.tmc_seed, args.file_path, args.prompt, args.approximate,
                        args.run_shapley, args.per_point, args.posion, args.signgd, args.early_stopping)
    for p in mp.active_children():
        p.terminate()