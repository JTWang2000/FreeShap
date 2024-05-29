import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import threading
import time

from .ntk import load_ntk, save_ntk


################################### Kernel regression with Dynamic Programming INVerse ################################
class shapleyNTKRegression(nn.Module):
    def __init__(self, k_train, y, n_class, pre_inv=None):
        """
        Parameters
        ----------
        k_train
        y
        n_class
        inv
        batch_size: checkpoint when less than this compute the exact inverse
                                    greater than this compute the exact inverse only at this position
        """
        super(shapleyNTKRegression, self).__init__()
        self.y = y.double()
        self.n_class = n_class
        n = k_train.size(1)
        identity = torch.eye(n, device=k_train.device).unsqueeze(0)
        reg = 1e-6
        self.k_train = k_train + identity * reg
        self.pre_inv = pre_inv
        self.single_kernel = False
        if k_train.size(0) == 1:
            self.single_kernel = True
        self.k_train = self.k_train.double()

    def forward(self, k_test, return_inv=False):
        preds = []
        for i in range(self.n_class):
            yi = torch.clone(self.y)
            yi[self.y==i] = 1
            yi[self.y!=i] = 0
            # compute inverse or use cached inverse
            if self.pre_inv is not None:
                # https://en.wikipedia.org/wiki/Block_matrix
                n = self.k_train.size(1)
                # Extract sub matrices
                B = self.k_train[0][:n - 1, n - 1:]
                C = self.k_train[0][n - 1:, :n - 1]
                D = self.k_train[0][n - 1:, n - 1:]

                A_inv = self.pre_inv
                D_minus_CA_invB = D - torch.mm(C, torch.mm(A_inv, B))

                P = torch.cat((
                    torch.cat((A_inv + torch.mm(torch.mm(A_inv, B),
                                                torch.mm(torch.inverse(D_minus_CA_invB), torch.mm(C, A_inv))),
                               -torch.mm(A_inv, torch.mm(B, torch.inverse(D_minus_CA_invB)))), dim=1),
                    torch.cat((-torch.mm(torch.mm(torch.inverse(D_minus_CA_invB), C), A_inv),
                               torch.inverse(D_minus_CA_invB)), dim=1)
                ), dim=0)
                beta_i = P @ yi
            else:
                # solve linear systems
                if self.single_kernel:
                    beta_i = torch.linalg.solve(self.k_train[0], yi)
                else:
                    beta_i = torch.linalg.solve(self.k_train[i], yi)

            if self.single_kernel:
                pred_i = k_test[0].double() @ beta_i
            else:
                pred_i = k_test[i].double() @ beta_i
            preds.append(pred_i)
        y_pred = torch.stack(preds, dim=1)
        if return_inv:
            if self.pre_inv is not None:
                return y_pred, P
            else:
                if self.single_kernel:
                    return y_pred, torch.inverse(self.k_train[0])
                else:
                    return y_pred, torch.inverse(self.k_train[i])
        else:
            return y_pred


################################### Fast Kernel regression approximation with block-inverse ############################
class fastNTKRegression(nn.Module):
    def __init__(self, k_train, y, n_class, inv=None, batch_size=None):
        super(fastNTKRegression, self).__init__()
        self.y = y.double()
        self.n_class = n_class
        if inv is None:
            n = k_train.size(1)
            identity = torch.eye(n, device=k_train.device).unsqueeze(0)
            reg = 1e-6
            self.k_train = k_train + identity * reg
            self.inv = None
        else:
            self.inv = inv
        self.single_kernel = False
        if k_train.size(0) == 1:
            self.single_kernel = True
        self.k_train = self.k_train.double()
        self.batch_size = batch_size

    def forward(self, k_test):
        preds = []
        for i in range(self.n_class):
            yi = torch.clone(self.y)
            yi[self.y==i] = 1
            yi[self.y!=i] = 0
            # compute inverse or use cached inverse
            if self.inv is not None:
                beta_i = self.inv @ yi
            else:
                # solve linear systems
                if self.single_kernel:
                    if self.batch_size is not None:
                        beta_i = []
                        for j in range(0, self.k_train.size(1), self.batch_size):
                            H = self.k_train[0, j:j+self.batch_size, j:j+self.batch_size]
                            beta_j = torch.linalg.solve(H, yi[j:j + self.batch_size])
                            beta_i.append(beta_j)
                        beta_i = torch.cat(beta_i, dim=0)
                else:
                    # improvement notes: does not support multi kernel block diagonal but can be easily solved
                    beta_i = torch.linalg.solve(self.k_train[i], yi)
            if self.single_kernel:
                pred_i = k_test[0].double() @ beta_i
            else:
                pred_i = k_test[i].double() @ beta_i
            preds.append(pred_i)
        y_pred = torch.stack(preds, dim=1)
        return y_pred


################################### Standard Kernel regression ##############################################
class NTKRegression(nn.Module):
    def __init__(self, k_train, y, n_class, inv=None):
        super(NTKRegression, self).__init__()
        self.y = y.double()
        self.n_class = n_class
        if inv is None:
            n = k_train.size(1)
            identity = torch.eye(n, device=k_train.device).unsqueeze(0)
            reg = 1e-6
            self.k_train = k_train + identity * reg
            self.inv = None
        else:
            self.inv = inv
        self.single_kernel = False
        if k_train.size(0) == 1:
            self.single_kernel = True
        self.k_train = self.k_train.double()

    def forward(self, k_test):
        preds = []
        for i in range(self.n_class):
            yi = torch.clone(self.y) 
            yi[self.y==i] = 1
            yi[self.y!=i] = 0
            # compute inverse or use cached inverse
            if self.inv is not None:
                beta_i = self.inv @ yi 
            else:
                # solve linear systems
                if self.single_kernel:
                    try:
                        beta_i = torch.linalg.solve(self.k_train[0], yi)
                    except:
                        print("Singular matrix")
                        beta_i = torch.linalg.lstsq(self.k_train[0], yi.unsqueeze(1)).solution.squeeze()

                else:
                    try:
                        beta_i = torch.linalg.solve(self.k_train[i], yi)
                    except:
                        print("Singular matrix")
                        beta_i = torch.linalg.lstsq(self.k_train[i], yi.unsqueeze(1)).solution.squeeze()
            if self.single_kernel:
                pred_i = k_test[0].double() @ beta_i
            else:
                pred_i = k_test[i].double() @ beta_i
            preds.append(pred_i)
        y_pred = torch.stack(preds, dim=1)
        return y_pred


############################# Kernel regression with correction when multiple class ####################################
class NTKRegression_correction_multiclass(nn.Module):
    def __init__(self, k_train, y, n_class, train_logits, test_logits, inv=None):
        super(NTKRegression_correction_multiclass, self).__init__()
        self.y = y.double()
        self.n_class = n_class
        if inv is None:
            n = k_train.size(1)
            identity = torch.eye(n, device=k_train.device).unsqueeze(0)
            reg = 1e-6
            self.k_train = k_train + identity * reg
            self.inv = None
        else:
            self.inv = inv
        self.train_logits = train_logits
        self.test_logits = test_logits
        self.single_kernel = False
        self.k_train = self.k_train.double()
        if k_train.size(0) == 1:
            self.single_kernel = True

    def forward(self, k_test):
        preds = []
        for i in range(self.n_class):
            yi = torch.clone(self.y)
            yi[self.y==i] = 1
            yi[self.y!=i] = 0
            # add correction
            yi = yi - self.train_logits[:, i]
            if self.single_kernel:
                i = 0
            # compute inverse or use cached inverse
            if self.inv is not None:
                beta_i = self.inv @ yi
            else:
                try:
                    beta_i = torch.linalg.solve(self.k_train[i], yi)
                except:
                    print("Singular matrix")
                    beta_i = torch.linalg.lstsq(self.k_train[i], yi.unsqueeze(1)).solution.squeeze()

            pred_i = k_test[i].double() @ beta_i
            preds.append(pred_i)

        y_pred = torch.stack(preds, dim=1)
        try:
            y_pred = y_pred + self.test_logits
        except:
            pass
        return y_pred