from collections import defaultdict
from math import comb
from tqdm import tqdm
import numpy as np
from torch.multiprocessing import Pool, Process, set_start_method
from multiprocessing import Pool

from dvutils.Adpt_Shapley import Adpt_Shapley
from dvutils.utils import merge_dataloaders, powerset, softmax
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy

from utils import InitYAMLObject
import torch
import gc
import pickle
from copy import deepcopy
import time
import random
import math

def worker(subset, data_loader, val_loader, train_process, probe_model, gpu_id, null_score):
    """
    Worker function to be executed by each process.
    Evaluates a given subset on a specific GPU.
    """
    torch.cuda.set_device(gpu_id)

    # Assuming self.evaluate() and other necessary methods are adapted to be static or class methods
    if len(subset) != 0:
        # Set the current device to the specific GPU
        train_process = deepcopy(train_process)
        probe_model = deepcopy(probe_model)

        train_process.args['device'] = torch.device(f'cuda:{gpu_id}')
        probe_model.args['device'] = torch.device(f'cuda:{gpu_id}')
        probe_model.model.to(probe_model.args['device'])

        # Subset and merge dataloaders as necessary
        # Ensure that this operation is compatible with multiprocessing
        train_process.train_until_convergence(probe_model, data_loader, val_loader,
                                              gradient_steps_between_eval=20 * len(data_loader))
        new_v_entropy, new_acc = train_process.predict(probe_model, val_loader)
        score = np.array([-new_v_entropy, new_acc])
        gc.collect()
        # print("score:", subset, score)
        return frozenset(subset), score
    else:
        return frozenset(), null_score


########################################################## FreeShap #############################################
class Fast_Data_Shapley(Adpt_Shapley, InitYAMLObject):
    ''' Running data shapley based on NTK '''
    yaml_tag = '!Fast_Data_Shapley'

    def __init__(self, dataset, probe_model, num_metric):
        Adpt_Shapley.__init__(self, probe_model, None, None, None)
        self.dataset = dataset
        self.probe_model = probe_model
        self.sv_result = []
        self.mc_cache = []
        self.ac_cache = []
        self.num_metric = num_metric

        self.train_loader = None
        self.val_loader = None
        self.data_idx = None

        self.train_set = None
        self.val_set = None

        self.initialize_ntk_state = False

    def tmc_one_iteration_idx(self, target_idx=0, approximate=False):
        """Runs one iteration of TMC-Shapley algorithm to find shapley value for target_idx"""
        idxs = np.random.permutation(self.n_participants)
        # print(idxs[:10])

        # find position for target_idx (the target_idx is always the first one)
        target_pos = np.where(idxs == target_idx)[0][0]

        # V(S)
        if target_pos == 0:
            old_score = self.get_null_score()
        else:
            selected_idx = idxs[:target_pos]

            if approximate:
                new_v_entropy, new_acc = self.probe_model.kernel_regression_idx(selected_idx, self.val_set,
                                                                                has_pre_inv=False)
            else:
                new_v_entropy, new_acc = self.probe_model.kernel_regression(selected_idx, self.val_set)
            old_score = np.array([-new_v_entropy, new_acc])

        with torch.cuda.device(self.probe_model.args['device']):
            torch.cuda.empty_cache()
        gc.collect()

        # V(S U {i})
        selected_idx = idxs[:target_pos + 1]
        if approximate:
            new_v_entropy, new_acc = self.probe_model.kernel_regression_idx(selected_idx, self.val_set, has_pre_inv=True)
        else:
            new_v_entropy, new_acc = self.probe_model.kernel_regression(selected_idx, self.val_set)
        new_score = np.array([-new_v_entropy, new_acc])

        with torch.cuda.device(self.probe_model.args['device']):
            torch.cuda.empty_cache()
        gc.collect()

        marginal_contribs = new_score - old_score

        def _tmc_compute(marginal_contribs):
            for metric_idx in range(self.num_metric):
                self.sv_result[metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[metric_idx]

        self.mc_cache.append(np.copy(marginal_contribs))
        _tmc_compute(marginal_contribs=marginal_contribs)

    def tmc_one_iteration_debug(self):
        """One iteration of TMC-Shapley algorithm and mainly to check the marginal contribution"""
        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros([self.n_participants, self.num_metric])

        full_score = self.get_full_score()
        print(full_score)

        new_score = self.get_null_score()
        print(new_score)
        selected_idx = []
        tmp_inspect = np.zeros([self.n_participants, self.num_metric])
        for n, idx in tqdm(enumerate(idxs), leave=False):
            old_score = new_score
            selected_idx.append(idx)
            new_v_entropy, new_acc = self.probe_model.kernel_regression(np.array(selected_idx), self.val_set)
            new_score = np.array([-new_v_entropy, new_acc])

            tmp_inspect[n] = new_score
            marginal_contribs[idx] = (new_score - old_score)

        return tmp_inspect, marginal_contribs


    def tmc_one_iteration(self, early_stopping=False, tolerance=0.05):
        """One iteration of FreeShap algorithm"""
        def _tmc_compute(idxs, marginal_contribs):
            for metric_idx in range(self.num_metric):
                for idx in idxs:
                    self.sv_result[idx][metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[idx][metric_idx]

        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros([self.n_participants, self.num_metric])

        truncation_counter = 0
        new_score = self.get_null_score()
        selected_idx = []
        tmp_inspect = np.zeros([self.n_participants, self.num_metric])
        for n, idx in tqdm(enumerate(idxs), leave=False):
            old_score = new_score
            selected_idx.append(idx)

            new_v_entropy, new_acc = self.probe_model.kernel_regression(np.array(selected_idx), self.val_set)
            new_score = np.array([-new_v_entropy, new_acc])

            tmp_inspect[n] = new_score
            marginal_contribs[idx] = (new_score - old_score)
            if early_stopping:
                distance_to_full_score = np.abs(new_score - self.get_full_score())
                if (distance_to_full_score <= tolerance * np.abs(self.get_full_score() - self.get_null_score())).all():
                    truncation_counter += 1
                    if truncation_counter > 1:
                        print(n)
                        break
                else:
                    truncation_counter = 0

        self.mc_cache.append(np.copy(marginal_contribs))
        self.ac_cache.append(np.copy(tmp_inspect))
        _tmc_compute(idxs=idxs, marginal_contribs=marginal_contribs)
        self.probe_model.pre_inv = None
        return idxs, marginal_contribs

    def tmc_one_iteration_per_point(self, early_stopping=False, tolerance=0.05):
        """One iteration of FreeShap algorithm and enables to get instance-level shapley value"""
        def _tmc_compute(idxs, marginal_contribs):
            for metric_idx in range(self.num_metric):
                for idx in idxs:
                    self.sv_result[idx][metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[idx][metric_idx]

        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros([self.n_participants, self.num_metric, len(self.val_set)])

        # truncation_counter = 0
        new_score = self.get_null_score_per_point()
        null_score_ = np.array([new_score[0].sum()/len(self.val_set), new_score[1].sum()/len(self.val_set)])
        selected_idx = []
        for n, idx in enumerate(idxs):
            old_score = new_score
            selected_idx.append(idx)

            new_v_entropy, new_acc = self.probe_model.kernel_regression(np.array(selected_idx), self.val_set,
                                                                        per_point=True)
            new_score = np.array([-new_v_entropy, new_acc.numpy()])
            marginal_contribs[idx] = (new_score - old_score)


            if early_stopping:
                new_score_ = np.array([new_score[0].sum()/len(self.val_set), new_score[1].sum()/len(self.val_set)])
                # print(new_score_)
                distance_to_full_score = np.abs(new_score_ - self.get_full_score_per_point())
                if (distance_to_full_score <= tolerance * np.abs(self.get_full_score_per_point() - null_score_)).all():
                    truncation_counter += 1
                    if truncation_counter > 1:
                        # print(n)
                        break
                else:
                    truncation_counter = 0

        # self.mc_cache.append(np.copy(marginal_contribs))
        _tmc_compute(idxs=idxs, marginal_contribs=marginal_contribs)
        self.probe_model.pre_inv = None

    def get_null_score(self):
        """To compute the performance with initial weight"""
        try:
            self.null_score
        except:
            # If the trainset is empty, ntk cannot be used. Thus, we adopt random guessing as the null score.
            num_labels = self.probe_model.num_labels
            acc = 1.0 / num_labels
            v_entropy = -num_labels * acc * np.log(acc)
            # improvement idea: might need to adjust this based on the actual distribution of the dataset (num_instances in each class)
            self.null_score = np.array([-v_entropy, acc])
        return self.null_score

    def get_full_score(self):
        """To compute the performance on grand coalition"""
        try:
            self.full_score
        except:
            selected_idx = np.arange(self.n_participants)
            v_entropy, acc = self.probe_model.kernel_regression(selected_idx, self.val_set)
            self.full_score = np.array([-v_entropy, acc])
        self.probe_model.pre_inv = None
        return self.full_score

    def get_null_score_per_point(self):
        """To compute the performance with initial weight"""
        try:
            self.null_score
        except:
            # If the trainset is empty, ntk cannot be used. Thus, we adopt random guessing as the null score.
            num_labels = self.probe_model.num_labels
            acc = 1.0 / num_labels
            v_entropy = -num_labels * acc * np.log(acc)

            self.null_score = np.array([np.full((len(self.val_set),), -v_entropy),
                                        np.full((len(self.val_set),), acc)])
        return self.null_score

    def get_full_score_per_point(self):
        """To compute the performance on grand coalition"""
        try:
            self.full_score
        except:
            selected_idx = np.arange(self.n_participants)
            v_entropy, acc = self.probe_model.kernel_regression(selected_idx, self.val_set, per_point=True)
            self.full_score = np.array([-v_entropy, acc])
        return self.full_score

    def initialize_ntk(self):
        # Given that train_loader and val_loader are provided in run(), prepare datasets
        # Set parameters for ntk computation
        # compute ntk matrix
        print("---------------------------initialize ntk-------------------------------")
        self.probe_model.compute_ntk(self.train_set, self.val_set)
        self.initialize_ntk_state = True

    def debug(self,
              data_idx,
              val_data_idx):
        """Compute the sv with different method"""
        self.mc_cache = []

        train_set = self.dataset.get_idx_dataset(data_idx, split="train")
        val_set = self.dataset.get_idx_dataset(val_data_idx, split="val")

        self.train_set = train_set
        self.val_set = val_set
        self.data_idx = data_idx
        self.n_participants = len(data_idx)

        # prepare the ntk matrix for the full dataset
        self.initialize_ntk()

        return self.tmc_one_iteration_debug()

    def run(self,
            data_idx,
            val_data_idx,
            iteration=1000,
            method="tmc",
            metric="accu",
            use_cache_ntk=False,
            prompt=True,
            seed=2023,
            num_dp=5000,
            checkpoint=False,
            per_point=False,
            early_stopping=False,
            tolerance=0.05):
        """Compute the sv with different method"""
        print("start to compute shapley value")
        if seed != 2023:
            np.random.seed(seed)
        self.data_idx = data_idx
        self.n_participants = len(data_idx)
        self.tmc_iteration = iteration
        self.sv_result = np.zeros([self.n_participants, self.num_metric])
        if per_point:
            self.sv_result = np.zeros([self.n_participants, self.num_metric, len(val_data_idx)])
        self.mc_cache = []

        train_set = self.dataset.get_idx_dataset(data_idx, split="train")
        val_set = self.dataset.get_idx_dataset(val_data_idx, split="val")

        self.train_set = train_set
        self.val_set = val_set

        self.metric = metric
        if method == "tmc":
            # prepare the ntk matrix for the full dataset
            if not use_cache_ntk:
                self.initialize_ntk()
            full_score = self.get_full_score()
            # print(f"full_score: {full_score}")
            # run tmc iterations based on the ntk matrix
            for curr_iter in tqdm(range(iteration), desc='[TMC iterations]'):
                # self.tmc_one_iteration_idx()
                if per_point:
                    self.tmc_one_iteration_per_point(early_stopping=early_stopping)
                else:
                    self.tmc_one_iteration(early_stopping=early_stopping, tolerance=tolerance)
                    if checkpoint and curr_iter % 20 == 0:
                        with open(f"../cache/{self.dataset.data_loader.dataset_name}_shapley_cache_{prompt}_seed{seed}_num{num_dp}_tolerance{tolerance}.pkl", "wb") as f:
                            pickle.dump(self.mc_cache, f)
                            pickle.dump(self.ac_cache, f)
            sv_result = self.sv_result
        elif method == "exact":
            self.exact_method(metric)
            sv_result = self.exact_sv_from_mem()
        return sv_result

    def run_idx_init(self,
                     data_idx,
                     val_data_idx,
                     method="tmc"):
        """Initialize NTK kernel and dataset"""
        self.data_idx = data_idx
        self.n_participants = len(data_idx)
        train_set = self.dataset.get_idx_dataset(data_idx, split="train")
        val_set = self.dataset.get_idx_dataset(val_data_idx, split="val")

        self.train_set = train_set
        self.val_set = val_set

        if method == "tmc":
            # prepare the ntk matrix for the full dataset
            self.initialize_ntk()
            full_score = self.get_full_score()
            print(f"full_score: {full_score}")

    def run_idx_init_dataset_only(self,
                                  data_idx,
                                  val_data_idx):
        """Initialize dataset"""
        self.data_idx = data_idx
        self.n_participants = len(data_idx)
        train_set = self.dataset.get_idx_dataset(data_idx, split="train")
        val_set = self.dataset.get_idx_dataset(val_data_idx, split="val")

        self.train_set = train_set
        self.val_set = val_set
        self.initialize_ntk_state = True

    def run_idx(self,
                target_idx,
                data_idx,
                val_data_idx,
                iteration=1000,
                method="tmc",
                metric="accu",
                seed=2023,
                approximate=False):
        """Compute the SV for target_idx"""
        # set seed
        np.random.seed(seed)
        self.tmc_iteration = iteration
        self.sv_result = np.zeros([self.num_metric])
        self.metric = metric
        self.mc_cache = []
        if not self.initialize_ntk_state:
            self.data_idx = data_idx
            self.n_participants = len(data_idx)

            train_set = self.dataset.get_idx_dataset(data_idx, split="train")
            val_set = self.dataset.get_idx_dataset(val_data_idx, split="val")

            self.train_set = train_set
            self.val_set = val_set

        if method == "tmc":
            # prepare the ntk matrix for the full dataset
            if not self.initialize_ntk_state:
                self.initialize_ntk()
                full_score = self.get_full_score()
                print(f"full_score: {full_score}")
            # run tmc iterations based on the ntk matrix
            for _ in tqdm(range(iteration), desc='[TMC iterations]'):
                self.tmc_one_iteration_idx(target_idx, approximate=approximate)
            sv_result = self.sv_result
        elif method == "exact":
            self.exact_method(metric)
            sv_result = self.exact_sv_from_mem()
        return sv_result


########################################################## Standard Shapley ###########################################

class Data_Shapley(Adpt_Shapley, InitYAMLObject):
    yaml_tag = '!Data_Shapley'

    def __init__(self, dataset, train_process, probe_model, num_metric):
        Adpt_Shapley.__init__(self, probe_model, None, None, None)
        self.dataset = dataset
        self.train_process = train_process
        self.probe_model = probe_model
        self.sv_result = []
        self.mc_cache = []
        self.init_model_weight = deepcopy(self.probe_model.state_dict())
        self.num_metric = num_metric

        self.train_loader = None
        self.val_loader = None
        self.data_idx = None

    def init(self):
        self.init_model_weight = deepcopy(self.probe_model.state_dict())

    def exact_method_parallel(self):
        all_subset = list(powerset(range(self.n_participants)))
        n_gpus = torch.cuda.device_count()

        def _balanced_subsets(all_subset, n_gpus):
            """
            Distributes subsets to GPUs in a balanced manner.
            Assumes the initial list is in increasing order of computational load.
            """
            # Shuffle the subsets to distribute computational load more evenly
            # This step can be adjusted based on a better understanding of the load distribution
            shuffled_subsets = sorted(all_subset, key=lambda x: (len(x), np.random.random()))

            # Now distribute these shuffled subsets evenly across GPUs
            subsets_per_gpu = [shuffled_subsets[i::n_gpus] for i in range(n_gpus)]

            return subsets_per_gpu

        # Usage
        subsets_per_gpu = _balanced_subsets(list(all_subset), n_gpus)

        # Create a pool of workers for each GPU
        pools = [torch.multiprocessing.Pool(processes=1) for _ in range(n_gpus)]

        results = []
        self.restart_model()
        for gpu_id, subsets in enumerate(subsets_per_gpu):
            # Use apply_async or map_async to dispatch the work to the pool
            # Each pool has only one process, effectively assigning each process to a different GPU
            for subset in tqdm(subsets):
                tmp_loader = self.dataset.get_idx_dataloader_reindx(list(subset), split="train")
                # self.restart_model()
                result = pools[gpu_id].apply_async(worker, (subset, tmp_loader, self.val_loader,
                                                            self.train_process, self.probe_model,
                                                            gpu_id, self.get_null_score()))
                results.append(result)

        # Close and join pools
        for pool in pools:
            pool.close()
        for pool in pools:
            pool.join()

        # Collect results
        score_for_allset = defaultdict()
        for result in results:
            subset, score = result.get()
            score_for_allset[subset] = score

        for i in range(self.n_participants):
            set_minus_i = [tmp for tmp in range(self.n_participants) if tmp != i]
            all_subset_i = powerset(set_minus_i)
            for subset in all_subset_i:
                subset_plus_i = list(subset) + [i]
                marginal_contribs = score_for_allset[frozenset(subset_plus_i)] - score_for_allset[frozenset(subset)]
                self.exact_record(frozenset(subset), i, marginal_contribs)

    def exact_method(self):
        """Exact method for estimating the adaptive sv"""
        all_subset = powerset(range(self.n_participants))
        score_for_allset = defaultdict()
        for subset in tqdm(all_subset):
            if len(subset) != 0:
                tmp_loader = self.dataset.get_idx_dataloader_reindx(list(subset), split="train")

                self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                           self.val_loader,
                                                           gradient_steps_between_eval=20*len(tmp_loader))
                new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader)
                score = np.array([-new_v_entropy, new_acc])
                score_for_allset[frozenset(subset)] = score
            score_for_allset[frozenset()] = self.get_null_score()
            self.restart_model()

        for i in range(self.n_participants):
            set_minus_i = [tmp for tmp in range(self.n_participants) if tmp != i]
            all_subset_i = powerset(set_minus_i)
            for subset in all_subset_i:
                subset_plus_i = list(subset) + [i]
                marginal_contribs = score_for_allset[frozenset(subset_plus_i)] - score_for_allset[frozenset(subset)]
                self.exact_record(frozenset(subset), i, marginal_contribs)

    def exact_record(self, coalition, idx, marginal_contribs):
        """Push the intermidiate result to memory for exact method"""
        try:
            self.memory[coalition][idx] = marginal_contribs
        except:
            self.memory[coalition] = defaultdict()
            self.memory[coalition][idx] = marginal_contribs

    def exact_sv_from_mem(self):
        """To compute the sv with exact algo from memory"""
        gamma_vec = [1/comb(self.n_participants-1, k) for k in range(self.n_participants)]

        sv_list = defaultdict(list)
        for coalition in self.memory:
            if len(coalition) != self.n_participants:
                for i, idx in enumerate(self.memory[coalition]):
                    set_size = len(coalition)
                    if set_size < self.n_participants:
                        weighted_marginal = self.memory[coalition][idx] * gamma_vec[set_size] * (1/self.n_participants)
                        sv_list[idx].append(weighted_marginal)
        sv_result = []
        for i in range(self.n_participants):
            sv_result.append(np.sum(np.array(sv_list[i]), axis=0))
        self.sv_result = sv_result
        return sv_result

    def tmc_one_iteration(self, tolerance=0.05, early_stopping=True):
        """One iteration of TMC-Shapley"""

        def _tmc_compute(idxs, marginal_contribs):
            for metric_idx in range(self.num_metric):
                for idx in idxs:
                    self.sv_result[metric_idx][idx] += (1.0 / self.tmc_iteration) * marginal_contribs[idx][
                        metric_idx]

        """Runs one iteration of TMC-Shapley algorithm"""
        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros([self.n_participants, self.num_metric])

        full_score = self.get_full_score()
        print(full_score)

        truncation_counter = 0
        new_score = self.get_null_score()
        _selected_idx = []
        tmp_inspect = np.zeros([self.n_participants, self.num_metric])
        for n, idx in tqdm(enumerate(idxs), leave=False):
            old_score = new_score
            _selected_idx.append(idx)
            tmp_loader = self.dataset.get_idx_dataloader_reindx(_selected_idx, split="train")
            self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                       self.val_loader,
                                                       gradient_steps_between_eval=5 * len(tmp_loader))
            new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader)
            new_score = np.array([-new_v_entropy, new_acc])

            tmp_inspect[n] = new_score
            marginal_contribs[idx] = (new_score - old_score)

            self.restart_model()

            if early_stopping:
                distance_to_full_score = np.abs(new_score - self.get_full_score())
                if (distance_to_full_score <= tolerance * np.abs(
                        self.get_full_score() - self.get_null_score())).all():
                    truncation_counter += 1
                    if truncation_counter > 1:
                        print(n)
                        break
                else:
                    truncation_counter = 0
        self.mc_cache.append(np.copy(marginal_contribs))
        _tmc_compute(idxs=idxs, marginal_contribs=marginal_contribs)

    def tmc_one_iteration_debug(self):
        """One iteration of TMC-Shapley algorithm and mainly to check the marginal contribution"""
        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros([self.n_participants, self.num_metric])

        full_score = self.get_full_score()
        print(full_score)
        new_score = self.get_null_score()
        print(new_score)
        _selected_idx = []
        tmp_inspect = np.zeros([self.n_participants, self.num_metric])
        for n, idx in tqdm(enumerate(idxs), leave=False):
            old_score = new_score
            _selected_idx.append(idx)
            tmp_loader = self.dataset.get_idx_dataloader_reindx(_selected_idx, split="train")
            self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                       self.val_loader, gradient_steps_between_eval=len(tmp_loader))
            new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader)
            new_score = np.array([-new_v_entropy, new_acc])

            tmp_inspect[n] = new_score
            marginal_contribs[idx] = (new_score - old_score)
            print(len(_selected_idx), new_score)

            self.restart_model()

        return tmp_inspect, marginal_contribs

    def tmc_one_iteration_mc(self, step=1, target_idx=0):
        """put target_idx in every collation and compute its marginal contribution, mainly for ablation study"""
        idxs = np.random.permutation(self.n_participants - 1) + 1
        assert target_idx not in idxs
        marginal_contribs = np.zeros([math.ceil(self.n_participants / step)+1, self.num_metric])
        tmp_inspect_without = np.zeros([math.ceil(self.n_participants / step)+1, self.num_metric])
        tmp_inspect_with = np.zeros([math.ceil(self.n_participants / step)+1, self.num_metric])

        full_score = self.get_full_score()
        print(full_score)
        old_score = self.get_null_score()
        print(old_score)

        _selected_idx = [target_idx]
        tmp_loader = self.dataset.get_idx_dataloader_reindx(_selected_idx, split="train")
        self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                   self.val_loader, gradient_steps_between_eval=20 * len(tmp_loader))
        new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader)
        new_score = np.array([-new_v_entropy, new_acc])
        marginal_contribs[0] = (new_score - old_score)
        tmp_inspect_without[0] = old_score
        tmp_inspect_with[0] = new_score
        self.restart_model()

        n = 0
        for i in tqdm(range(0, self.n_participants, step)):
            _selected_idx = idxs[:i + 1]
            # without target_idx
            tmp_loader = self.dataset.get_idx_dataloader_reindx(_selected_idx, split="train")
            self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                       self.val_loader, gradient_steps_between_eval=20 * len(tmp_loader))
            old_v_entropy, old_acc = self.train_process.predict(self.probe_model, self.val_loader)
            old_score = np.array([-old_v_entropy, old_acc])
            self.restart_model()

            # with target_idx
            _selected_idx = _selected_idx.tolist() + [target_idx]
            tmp_loader = self.dataset.get_idx_dataloader_reindx(_selected_idx, split="train")
            self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                       self.val_loader, gradient_steps_between_eval=5 * len(tmp_loader))
            new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader)
            new_score = np.array([-new_v_entropy, new_acc])
            # print(old_score, new_score)
            marginal_contribs[n + 1] = (new_score - old_score)
            tmp_inspect_without[n + 1] = old_score
            tmp_inspect_with[n + 1] = new_score

            self.restart_model()
            n += 1

        return tmp_inspect_with, tmp_inspect_without, marginal_contribs

    def tmc_one_iteration_idx_per_point(self, target_idx=0):
        """One iteration of TMC-Shapley to find shapley value for idx, mainly for robust/correlation experiment"""
        idxs = np.random.permutation(self.n_participants)

        # find position for target_idx
        target_pos = np.where(idxs == target_idx)[0][0]

        self.restart_model()

        # V(S)
        if target_pos == 0:
            old_score = self.get_null_score_per_point()
        else:
            selected_idx = idxs[:target_pos]
            tmp_loader = self.dataset.get_idx_dataloader_reindx(selected_idx, split="train")
            self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                       self.val_loader, gradient_steps_between_eval=5 * len(tmp_loader))
            new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader, per_point=True)
            old_score = np.array([-new_v_entropy, new_acc])

        self.restart_model()
        with torch.cuda.device(self.probe_model.args['device']):
            torch.cuda.empty_cache()
        gc.collect()

        # V(S U {i})
        selected_idx = idxs[:target_pos + 1]
        tmp_loader = self.dataset.get_idx_dataloader_reindx(selected_idx, split="train")
        self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                   self.val_loader, gradient_steps_between_eval=5 * len(tmp_loader))
        new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader, per_point=True)
        new_score = np.array([-new_v_entropy, new_acc])
        with torch.cuda.device(self.probe_model.args['device']):
            torch.cuda.empty_cache()
        gc.collect()

        marginal_contribs = new_score - old_score

        def _tmc_compute(marginal_contribs):
            for metric_idx in range(self.num_metric):
                self.sv_result[metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[metric_idx]

        # self.mc_cache.append(np.copy(marginal_contribs))
        _tmc_compute(marginal_contribs=marginal_contribs)

    def get_null_score_per_point(self):
        """To compute the performance with initial weight"""
        try:
            self.null_score
        except:
            self.restart_model()
            v_entropy, acc = self.train_process.predict(self.probe_model, self.val_loader, per_point=True)
            self.null_score = np.array([-v_entropy, acc])
        return self.null_score

    def tmc_one_iteration_idx(self, target_idx=0):
        """Runs one iteration of TMC-Shapley algorithm to find shapley value for idx"""
        idxs = np.random.permutation(self.n_participants)
        # print(idxs[:10])

        # find position for target_idx
        target_pos = np.where(idxs == target_idx)[0][0]

        self.restart_model()

        # V(S)
        if target_pos == 0:
            old_score = self.get_null_score()
        else:
            selected_idx = idxs[:target_pos]
            tmp_loader = self.dataset.get_idx_dataloader_reindx(selected_idx, split="train")
            self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                       self.val_loader, gradient_steps_between_eval=5 * len(tmp_loader))
            new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader)
            old_score = np.array([-new_v_entropy, new_acc])

        self.restart_model()
        with torch.cuda.device(self.probe_model.args['device']):
            torch.cuda.empty_cache()
        gc.collect()

        # V(S U {i})
        selected_idx = idxs[:target_pos + 1]
        tmp_loader = self.dataset.get_idx_dataloader_reindx(selected_idx, split="train")
        self.train_process.train_until_convergence(self.probe_model, tmp_loader,
                                                   self.val_loader, gradient_steps_between_eval=5 * len(tmp_loader))
        new_v_entropy, new_acc = self.train_process.predict(self.probe_model, self.val_loader)
        new_score = np.array([-new_v_entropy, new_acc])
        with torch.cuda.device(self.probe_model.args['device']):
            torch.cuda.empty_cache()
        gc.collect()

        marginal_contribs = new_score - old_score
        print(old_score[1], new_score[1], marginal_contribs[1])

        def _tmc_compute(marginal_contribs):
            for metric_idx in range(self.num_metric):
                self.sv_result[metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[metric_idx]

        self.mc_cache.append(np.copy(marginal_contribs))
        _tmc_compute(marginal_contribs=marginal_contribs)


    def get_null_score(self):
        """To compute the performance with initial weight"""
        try:
            self.null_score
        except:
            self.restart_model()
            v_entropy, acc = self.train_process.predict(self.probe_model, self.val_loader)
            self.null_score = np.array([-v_entropy, acc])
        return self.null_score

    def get_full_score(self):
        """To compute the performance on grand coalition"""
        try:
            self.full_score
        except:
            self.train_process.train_until_convergence(self.probe_model, self.train_loader, self.val_loader,
                                                       gradient_steps_between_eval=min(1000, len(self.train_loader)))
            v_entropy, acc = self.train_process.predict(self.probe_model, self.val_loader)
            self.full_score = np.array([-v_entropy, acc])
            self.restart_model()
        return self.full_score

    def restart_model(self):
        self.probe_model.load_state_dict(deepcopy(self.init_model_weight))

    def run_idx(self,
                target_idx,
                data_idx,
                val_data_idx,
                iteration=2000,
                metric="accu",
                seed=2023,
                per_point=False):
        """Compute the SV for target_idx, mainly used for robust/correlation experiment"""
        np.random.seed(seed)

        self.train_loader = self.dataset.get_idx_dataloader(data_idx, split="train")
        self.val_loader = self.dataset.get_idx_dataloader(val_data_idx, split="val")
        self.data_idx = data_idx
        self.n_participants = len(data_idx)
        self.tmc_iteration = iteration
        self.sv_result = np.zeros([self.num_metric])
        self.mc_cache = []
        self.metric = metric

        if per_point:
            self.sv_result = np.zeros([self.num_metric, len(val_data_idx)])

        for _ in tqdm(range(iteration), desc='[TMC iterations]'):
            if per_point:
                self.tmc_one_iteration_idx_per_point(target_idx=target_idx)
            else:
                self.tmc_one_iteration_idx(target_idx)
        sv_result = self.sv_result

        return sv_result

    def debug(self,
              data_idx,
              val_data_idx):
        """One iteration of MC, mainly for ablation study purpose"""

        self.train_loader = self.dataset.get_idx_dataloader(data_idx, split="train")
        self.val_loader = self.dataset.get_idx_dataloader(val_data_idx, split="val")
        self.data_idx = data_idx
        self.n_participants = len(data_idx)

        # return self.tmc_one_iteration_debug()
        return self.tmc_one_iteration_mc(10)

    def run(self,
            data_idx,
            val_data_idx,
            iteration=1000,
            method="tmc",
            metric="accu",
            seed=2023,
            early_stopping=False):

        """Compute the SV for data_idx"""
        np.random.seed(seed)

        self.train_loader = self.dataset.get_idx_dataloader(data_idx, split="train")
        self.val_loader = self.dataset.get_idx_dataloader(val_data_idx, split="val")
        self.data_idx = data_idx
        self.n_participants = len(data_idx)
        self.tmc_iteration = iteration
        self.sv_result = np.zeros([self.num_metric, self.n_participants])
        self.mc_cache = []
        self.metric = metric


        if method == "tmc":
            full_score = self.get_full_score()
            print(f"full_score: {full_score}")
            for _ in tqdm(range(iteration), desc='[TMC iterations]'):
                self.tmc_one_iteration(early_stopping=early_stopping)
            sv_result = self.sv_result
        elif method == "exact":
            if torch.cuda.device_count() > 1:
                self.exact_method_parallel()
            else:
                self.exact_method()
            sv_result = self.exact_sv_from_mem()
        return sv_result


########################################################## Linear Kernel Shapley #############################################
class Fast_Linear_Data_Shapley(Adpt_Shapley, InitYAMLObject):
    ''' Running data shapley based on last layer activation kernel '''
    yaml_tag = '!Fast_Linear_Data_Shapley'

    def __init__(self, dataset, probe_model, num_metric):
        Adpt_Shapley.__init__(self, probe_model, None, None, None)
        self.dataset = dataset
        self.probe_model = probe_model
        self.sv_result = []
        self.mc_cache = []
        self.ac_cache = []
        self.num_metric = num_metric

        self.train_loader = None
        self.val_loader = None
        self.data_idx = None

        self.train_set = None
        self.val_set = None

        self.initialize_ntk_state = False

    def tmc_one_iteration(self, early_stopping=False, tolerance=0.05):
        """Runs one iteration of TMC-Shapley algorithm"""
        def _tmc_compute(idxs, marginal_contribs):
            for metric_idx in range(self.num_metric):
                for idx in idxs:
                    self.sv_result[idx][metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[idx][metric_idx]

        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros([self.n_participants, self.num_metric])

        truncation_counter = 0
        new_score = self.get_null_score()
        selected_idx = []
        tmp_inspect = np.zeros([self.n_participants, self.num_metric])
        for n, idx in tqdm(enumerate(idxs), leave=False):
            old_score = new_score
            selected_idx.append(idx)

            new_v_entropy, new_acc = self.probe_model.kernel_regression(np.array(selected_idx), self.val_set)
            new_score = np.array([-new_v_entropy, new_acc])

            tmp_inspect[n] = new_score
            marginal_contribs[idx] = (new_score - old_score)
            if early_stopping:
                distance_to_full_score = np.abs(new_score - self.get_full_score())
                if (distance_to_full_score <= tolerance * np.abs(self.get_full_score() - self.get_null_score())).all():
                    truncation_counter += 1
                    if truncation_counter > 1:
                        print(n)
                        break
                else:
                    truncation_counter = 0

        self.mc_cache.append(np.copy(marginal_contribs))
        self.ac_cache.append(np.copy(tmp_inspect))
        _tmc_compute(idxs=idxs, marginal_contribs=marginal_contribs)
        self.probe_model.pre_inv = None
        return idxs, marginal_contribs

    def tmc_one_iteration_idx(self, target_idx=0):
        """Runs one iteration of TMC-Shapley algorithm to find shapley value for idx"""
        idxs = np.random.permutation(self.n_participants)
        # print(idxs[:10])

        # find position for target_idx (the target_idx is always the first one)
        target_pos = np.where(idxs == target_idx)[0][0]

        # V(S)
        if target_pos == 0:
            old_score = self.get_null_score()
        else:
            selected_idx = idxs[:target_pos]

            new_v_entropy, new_acc = self.probe_model.kernel_regression(selected_idx, self.val_set)
            old_score = np.array([-new_v_entropy, new_acc])

        with torch.cuda.device(self.probe_model.args['device']):
            torch.cuda.empty_cache()
        gc.collect()

        # V(S U {i})
        selected_idx = idxs[:target_pos + 1]
        new_v_entropy, new_acc = self.probe_model.kernel_regression(selected_idx, self.val_set)
        new_score = np.array([-new_v_entropy, new_acc])

        with torch.cuda.device(self.probe_model.args['device']):
            torch.cuda.empty_cache()
        gc.collect()

        marginal_contribs = new_score - old_score

        def _tmc_compute(marginal_contribs):
            for metric_idx in range(self.num_metric):
                self.sv_result[metric_idx] += (1.0 / self.tmc_iteration) * marginal_contribs[metric_idx]

        self.mc_cache.append(np.copy(marginal_contribs))
        _tmc_compute(marginal_contribs=marginal_contribs)

    def get_null_score(self):
        """To compute the performance with initial weight"""
        try:
            self.null_score
        except:
            # If the trainset is empty, ntk cannot be used. Thus, we adopt random guessing as the null score.
            num_labels = self.probe_model.num_labels
            acc = 1.0 / num_labels
            v_entropy = -num_labels * acc * np.log(acc)
            self.null_score = np.array([-v_entropy, acc])
        return self.null_score

    def get_full_score(self):
        """To compute the performance on grand coalition"""
        try:
            self.full_score
        except:
            selected_idx = np.arange(self.n_participants)
            v_entropy, acc = self.probe_model.kernel_regression(selected_idx, self.val_set)
            self.full_score = np.array([-v_entropy, acc])
        self.probe_model.pre_inv = None
        return self.full_score

    def initialize_linear_kernel(self):
        # Given that train_loader and val_loader are provided in run(), prepare datasets
        # Set parameters for ntk computation
        # compute ntk matrix
        self.probe_model.compute_last_activation_kernel(self.train_loader, self.val_loader)
        self.initialize_linear_kernel_state = True

    def run_idx_init_linear_kernel(self,
                                   data_idx,
                                   val_data_idx,
                                   method="tmc"):
        self.data_idx = data_idx
        self.n_participants = len(data_idx)

        self.train_set = self.dataset.get_idx_dataset(data_idx, split="train")
        self.val_set = self.dataset.get_idx_dataset(val_data_idx, split="val")
        self.train_loader = DataLoader(self.train_set, batch_size=self.dataset.args['batchsize'], shuffle=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.dataset.args['batchsize'], shuffle=False)

        if method == "tmc":
            # prepare the ntk matrix for the full dataset
            self.initialize_linear_kernel()
            full_score = self.get_full_score()
            print(f"full_score: {full_score}")

    def run(self,
            data_idx,
            val_data_idx,
            iteration=1000,
            method="tmc",
            metric="accu",
            use_cache_ntk=False,
            seed=2023,
            early_stopping=False):
        """Compute the sv with different method"""
        np.random.seed(seed)
        self.data_idx = data_idx
        self.n_participants = len(data_idx)
        self.tmc_iteration = iteration
        # self.sv_result = np.zeros([self.num_metric])
        self.sv_result = np.zeros([self.n_participants, self.num_metric])
        if per_point:
            self.sv_result = np.zeros([self.n_participants, self.num_metric, len(val_data_idx)])
        self.mc_cache = []

        train_set = self.dataset.get_idx_dataset(data_idx, split="train")
        val_set = self.dataset.get_idx_dataset(val_data_idx, split="val")

        self.train_set = train_set
        self.val_set = val_set

        self.metric = metric
        if method == "tmc":
            # prepare the ntk matrix for the full dataset
            if not use_cache_ntk:
                self.initialize_ntk()
            full_score = self.get_full_score()
            print(f"full_score: {full_score}")
            # run tmc iterations based on the ntk matrix
            for _ in tqdm(range(iteration), desc='[TMC iterations]'):
                # self.tmc_one_iteration_idx()
                self.tmc_one_iteration(early_stopping=early_stopping)

            sv_result = self.sv_result
        elif method == "exact":
            self.exact_method(metric)
            sv_result = self.exact_sv_from_mem()
        return sv_result


    def run_idx(self,
                target_idx,
                data_idx,
                val_data_idx,
                iteration=1000,
                method="tmc",
                metric="accu",
                seed=2023):
        """Compute the sv with different method"""
        # set seed
        np.random.seed(seed)
        self.tmc_iteration = iteration
        # self.sv_result = np.zeros([self.n_participants, self.num_metric])
        self.sv_result = np.zeros([self.num_metric])
        self.metric = metric
        self.mc_cache = []
        if not self.initialize_linear_kernel_state:
            self.data_idx = data_idx
            self.n_participants = len(data_idx)

            train_set = self.dataset.get_idx_dataset(data_idx, split="train")
            val_set = self.dataset.get_idx_dataset(val_data_idx, split="val")

            self.train_set = train_set
            self.val_set = val_set

        if method == "tmc":
            # prepare the ntk matrix for the full dataset
            if not self.initialize_linear_kernel_state:
                self.initialize_ntk()
                full_score = self.get_full_score()
                print(f"full_score: {full_score}")
            # run tmc iterations based on the ntk matrix
            for _ in tqdm(range(iteration), desc='[TMC iterations]'):
                self.tmc_one_iteration_idx(target_idx)
            sv_result = self.sv_result
        elif method == "exact":
            self.exact_method(metric)
            sv_result = self.exact_sv_from_mem()
        return sv_result