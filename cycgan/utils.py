import torch
import random


class ReplayBuffer():
    def __init__(self, max_size=5):
        assert max_size > 0, 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class Lambda_LR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def decode_output(output):
    return [0]

import numpy as np
import pandas as pd
import sklearn

class InOutMapping:

    def __init__(self):
        self.binary_vars = {}
        self.expand_colnames = []
        self.mean = []
        self.sd = []

    # TODO only scale the numerical features
    def map_input(self, features, dummies):
        expand_features = pd.get_dummies(features, columns=dummies)
        for c in dummies:
            if features[c].nunique() == 2:
                self.binary_vars[c] = features[c].unique()
                # deleting one of the binary feature columns
                new_col = c + '_' + self.binary_vars[c][1]
                expand_features = expand_features.drop(columns=[new_col])
                # features = features.drop(columns=['sex_Male', 'Y_<=50K'])

        self.expand_colnames = expand_features.columns
        scaler = sklearn.preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(expand_features)
        self.mean = scaler.mean_
        self.sd = np.sqrt(scaler.var_)
        X = X_scaled.astype(np.float32)
        print(X.shape)
        return X

    # TODO revert the scaling for numerical and convert fload to int
    # TODO rescale everything for now
    def map_output(self, X, dummies, threshold=1):
        # TODO have the threshold
        onehot_cols = []
        noncat_cols_idx = []
        noncat_cols = []
        for idx, c in enumerate(self.expand_colnames):
            if any(map(c.startswith, dummies)):
                onehot_cols.append(c)  # TODO maybe I don't need this
            else:
                noncat_cols_idx.append(idx)
                noncat_cols.append(c)  # TODO reduce the code to a filter

        revert_df = pd.DataFrame(columns=dummies)
        for c in dummies:
            dummy_idx = [idx for idx, val in enumerate(self.expand_colnames) if val.startswith(c + '_')]
            corresponding = X[:, dummy_idx]
            if c in self.binary_vars.keys():
                b0 = self.binary_vars[c][0]
                b1 = self.binary_vars[c][1]
                values = [b1 if val < 0.5 else b0 for val in corresponding]  # TODO we don't have 0 and 1
            else:
                m = np.zeros_like(corresponding)
                m[np.arange(len(corresponding)), corresponding.argmax(1)] = 1
                max_col = np.argmax(m, axis=1)
                values = [self.expand_colnames[dummy_idx[i]].replace(c + '_', '') for i in max_col]
            revert_df[c] = values # TODO eliminate revert_df use res in

        res = pd.DataFrame(X[:, noncat_cols_idx], columns=noncat_cols)
        for c in dummies:
            res[c] = revert_df[c]
        for i in noncat_cols_idx:
            c = self.expand_colnames[i]
            res[c] = res[c] * self.sd[i] + self.mean[i]
        return res
        # TODO assert values are close
        # TODO fix the binary values
        # TODO tartibe sotuna moheme?

        # columns = expanded_X.columns
        # for dummy in list_dummies:
        #     get the columns startring with that-
        #     get the max of those columns
