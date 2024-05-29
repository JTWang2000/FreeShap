from collections import defaultdict
from math import comb

import numpy as np
from tqdm import tqdm

from dvutils.utils import merge_dataloaders, powerset, softmax


class Adpt_Shapley:
    def __init__(self, model, train_loaders, val_loader, n_participants):
        self.model = model
        self.train_loaders = train_loaders
        self.val_loader = val_loader
        self.n_participants = n_participants
        self.memory = defaultdict() # to store the intermidiate result
        self.adp_sv_result = [] # the adaptive sv result

    def evaluate(self, data_loader, metric="accu", early_stopping=True):
        """The evaluation of performance for a dataset"""
        self.model.restart_model()
        if data_loader != []:
            self.model.fit(data_loader, enable_early_stop=early_stopping)
        else:
            # null coalition
            pass
        loss, accu, f1_micro, f1_macro, f1_weighted = self.model.evaluate(self.val_loader)
        if metric == "accu":
            value_evaluated = accu
        elif metric == "f1-micro":
            value_evaluated = f1_micro
        elif metric == "f1-macro":
            value_evaluated = f1_macro
        elif metric == "f1-weighted":
            value_evaluated = f1_weighted
        else:
            raise "{} metric is not implememted".format(metric)
        return value_evaluated

    def tmc_record(self,idxs, marginal_contribs):
        """Push the intermidiate result to memory for tmc method"""
        for i in range(len(idxs)):
            coalition = frozenset(idxs[:i])
            try:
                self.memory[coalition][idxs[i]].append(marginal_contribs[idxs[i]])
            except:
                self.memory[coalition] = defaultdict(list)
                self.memory[coalition][idxs[i]].append(marginal_contribs[idxs[i]])

    def exact_record(self, coalition, idx, marginal_contribs):
        """Push the intermidiate result to memory for exact method"""
        try:
            self.memory[coalition][idx] = marginal_contribs
        except:
            self.memory[coalition] = defaultdict()
            self.memory[coalition][idx] = marginal_contribs

    def tmc_one_iteration(self, tolerance, metric, early_stopping=True):
        """Runs one iteration of TMC-Shapley algorithm."""
        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros(self.n_participants)
        truncation_counter = 0
        new_score = self.get_null_score()
        train_loaders = []
        for n, idx in tqdm(enumerate(idxs), leave=False):
        # for n, idx in enumerate(idxs):
            old_score = new_score
            train_loaders.append(self.train_loaders[idx])

            tmp_loader = merge_dataloaders(train_loaders)
            new_score = self.evaluate(tmp_loader, metric, early_stopping=early_stopping)  

            marginal_contribs[idx] = (new_score - old_score)
            distance_to_full_score = np.abs(new_score - self.get_full_score())
            if distance_to_full_score <= tolerance * self.get_full_score():
                truncation_counter += 1
                if truncation_counter > 1:
                    break
            else:
                truncation_counter = 0
        self.tmc_record(idxs=idxs,
                        marginal_contribs=marginal_contribs)


    def tmc_adp_sv_from_mem(self, temperature):
        """To compute the adaptive sv with TMC from memory with different hyper-parameters"""
        adp_sv_list = defaultdict(list)
        for coalition in self.memory:
            marginal_contibs = list(map(lambda x: np.mean(x), list(self.memory[coalition].values())))
            betas = softmax(marginal_contibs, temperature)
            for i, idx in enumerate(self.memory[coalition]):
                adp_sv_list[idx].extend(list(map(lambda x: betas[i]*x, self.memory[coalition][idx])))
        adp_sv_result = []
        for i in range(self.n_participants):
            adp_sv_result.append(np.mean(adp_sv_list[i]))
        self.adp_sv_result = adp_sv_result
        return adp_sv_result

    def exact_method(self, metric):
        """Exact method for estimating the adaptive sv"""
        all_subset = powerset(range(self.n_participants))
        score_for_allset = defaultdict()
        for subset in tqdm(all_subset):
            if len(subset) != 0:
                train_loaders = np.array(self.train_loaders)[list(subset)]
                train_loaders = merge_dataloaders(train_loaders)
                score = self.evaluate(train_loaders,metric)
                score_for_allset[frozenset(subset)] = score
            score_for_allset[frozenset()] = self.get_null_score()

        for i in range(self.n_participants):
            set_minus_i = [tmp for tmp in range(self.n_participants) if tmp != i]
            all_subset_i = powerset(set_minus_i)
            for subset in all_subset_i:
                subset_plus_i = list(subset) + [i]
                marginal_contribs = score_for_allset[frozenset(subset_plus_i)] - score_for_allset[frozenset(subset)]
                self.exact_record(frozenset(subset), i, marginal_contribs)
    
    def exact_adp_sv_from_mem(self, alpha, temperature, is_alpha):
        """To compute the adaptive sv with exact algo from memory with different hyper-parameters"""
        gamma_vec = [1/comb(self.n_participants-1, k) for k in range(self.n_participants)]
        def weighted_marginal_fn(x, beta, alpha, set_size):
            gamma = gamma_vec[set_size]
            weight = np.power(gamma, alpha) * np.power(beta, 1-alpha)
            return weight * x
        
        def weighted_marginal_noalpha_fn(x, beta, alpha, set_size):
            gamma = gamma_vec[set_size]
            weight = gamma * beta
            return weight * x

        adp_sv_list = defaultdict(list)
        for coalition in self.memory:
            if len(coalition) != self.n_participants:
                marginal_contibs = list(self.memory[coalition].values())
                betas = softmax(marginal_contibs, temperature)
                for i, idx in enumerate(self.memory[coalition]):
                    if is_alpha:
                        weighted_marginal = weighted_marginal_fn(self.memory[coalition][idx],
                                                             betas[i], alpha, len(coalition))
                    else:
                        weighted_marginal = weighted_marginal_noalpha_fn(self.memory[coalition][idx],
                                                             betas[i], alpha, len(coalition))

                    adp_sv_list[idx].append(weighted_marginal)
        adp_sv_result = []
        for i in range(self.n_participants):
            adp_sv_result.append(np.sum(adp_sv_list[i]))
        self.adp_sv_result = adp_sv_result
        return adp_sv_result
    
    def run(self,
            method="tmc",
            iteration=2000,  
            tolerance=0.01, 
            alpha = 0.5,
            temperature=2.0,
            metric="accu"):
        """Compute the adaptive sv with different method"""
        self.memory = defaultdict()
        self.metric = metric
        if method == "tmc":
            for iter in range(iteration):
                if 100*(iter+1)/iteration % 1 == 0:
                    print('{} out of {} TMC_Shapley iterations.'.format(
                        iter + 1, iteration))
                self.tmc_one_iteration(tolerance=tolerance, metric=metric)
            adp_sv_result = self.tmc_adp_sv_from_mem(temperature)
        elif method == "exact":
            self.exact_method(metric=metric)
            is_alpha = alpha >= 0
            adp_sv_result = self.exact_adp_sv_from_mem(alpha, temperature, is_alpha)
        return adp_sv_result

    def get_full_score(self):
        """To compute the performance on grand coalition"""
        try:
            self.full_score
        except:
            grand_train_loader = merge_dataloaders(self.train_loaders)
            
            # # debug
            # self.model.restart_model()
            # self.model.fit(grand_train_loader,epochs = 1000)
            # raise NotImplementedError
        
            self.full_score = self.evaluate(grand_train_loader,self.metric)
        return self.full_score

    def get_null_score(self):
        """To compute the performance with initial weight"""
        try:
            self.null_score
        except:
            null_train_loader = []
            self.null_score = self.evaluate(null_train_loader,self.metric)
        return self.null_score
