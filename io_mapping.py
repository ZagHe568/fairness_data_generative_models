import numpy as np
import pandas as pd
import sklearn.metrics


class InOutMapping:

    def __init__(self):
        self.binary_vars = {}
        self.onehot_colnames = []
        self.mean = []
        self.sd = []
        self.num_cols_idx = []
        self.num_cols = []
        self.ordered_columns = []

    def map_input(self, features, dummies):
        self.ordered_columns = features.columns
        for idx, c in enumerate(features.columns):
            if c not in dummies:
                #self.num_cols_idx.append(idx)
                self.num_cols.append(c)

        expand_features = pd.get_dummies(features[dummies], columns=dummies)
        # reduce the binary variables to only one column
        for c in dummies:
            if features[c].nunique() == 2:
                self.binary_vars[c] = features[c].unique()
                # deleting one of the binary feature columns
                new_col = c + '_' + self.binary_vars[c][1]
                expand_features = expand_features.drop(columns=[new_col])
                # features = features.drop(columns=['sex_Male', 'Y_<=50K'])


        self.onehot_colnames = expand_features.columns


        scaler = sklearn.preprocessing.StandardScaler()
        print(expand_features.shape)
        print(features[self.num_cols].shape)
        X = np.column_stack((features[self.num_cols].values, expand_features.values))
        print(X.shape)
        X_scaled = scaler.fit_transform(X)
        self.mean = scaler.mean_
        self.sd = np.sqrt(scaler.var_)
        print(X_scaled.shape)
        X_scaled = X_scaled.astype(np.float32)
        return X_scaled

    def map_output(self, X, dummies, threshold=1): # TODO have the threshold
        dummies_df = pd.DataFrame(columns=dummies)
        for c in dummies:
            dummy_idx = [len(self.num_cols)+idx for idx, val in enumerate(self.onehot_colnames) if val.startswith(c + '_')]
            corresponding = X[:, dummy_idx]
            if c in self.binary_vars.keys():
                b0 = self.binary_vars[c][0]
                b1 = self.binary_vars[c][1]
                values = [b1 if val < 0.5 else b0 for val in corresponding]
            else:
                m = np.zeros_like(corresponding)
                m[np.arange(len(corresponding)), corresponding.argmax(1)] = 1
                max_col = np.argmax(m, axis=1)
                values = [self.onehot_colnames[dummy_idx[i]-len(self.num_cols)].replace(c + '_', '') for i in max_col]
            dummies_df[c] = values  # TODO eliminate dummies_df use res in

        res = pd.DataFrame(X[:, 0:len(self.num_cols)], columns=self.num_cols)
        for c in dummies:
            res[c] = dummies_df[c]
        # scale back
        for i in range(len(self.num_cols)):
            c = self.num_cols[i]
            res[c] = int(res[c] * self.sd[i] + self.mean[i])
        # print(res.shape)
        # print(res.columns)
        reorder_res = pd.DataFrame()
        for c in self.ordered_columns:
            reorder_res[c] = res[c]
        print(reorder_res.shape)
        return reorder_res


class InOutMapping2:

    def __init__(self):
        self.binary_vars = {}
        self.onehot_colnames = []
        self.mean = []
        self.sd = []
        self.num_cols_idx = []
        self.num_cols = []
        self.ordered_columns = []

    def map_input(self, features, dummies):
        self.ordered_columns = features.columns
        for idx, c in enumerate(features.columns):
            if c not in dummies:
                #self.num_cols_idx.append(idx)
                self.num_cols.append(c)

        expand_features = pd.get_dummies(features[dummies], columns=dummies)
        print(expand_features.shape)
        # reduce the binary variables to only one column
        for c in dummies:
            if features[c].nunique() == 2:
                self.binary_vars[c] = features[c].unique()
                # deleting one of the binary feature columns
                new_col = c + '_' + self.binary_vars[c][1]
                expand_features = expand_features.drop(columns=[new_col])
                # features = features.drop(columns=['sex_Male', 'Y_<=50K'])

        print(expand_features.shape)
        self.onehot_colnames = expand_features.columns


        scaler = sklearn.preprocessing.StandardScaler()
        X_num_scaled = scaler.fit_transform(features[self.num_cols])
        self.mean = scaler.mean_
        self.sd = np.sqrt(scaler.var_)
        print(X_num_scaled.shape, expand_features.values.shape)
        X = np.column_stack((X_num_scaled, expand_features.values))
        X = X.astype(np.float32)
        print(X.shape)
        return X

    def map_output(self, X, dummies, threshold=1): # TODO have the threshold
        dummies_df = pd.DataFrame(columns=dummies)
        for c in dummies:
            dummy_idx = [len(self.num_cols)+idx for idx, val in enumerate(self.onehot_colnames) if val.startswith(c + '_')]
            corresponding = X[:, dummy_idx]
            if c in self.binary_vars.keys():
                b0 = self.binary_vars[c][0]
                b1 = self.binary_vars[c][1]
                values = [b1 if val < 0.5 else b0 for val in corresponding]
            else:
                m = np.zeros_like(corresponding)
                m[np.arange(len(corresponding)), corresponding.argmax(1)] = 1
                max_col = np.argmax(m, axis=1)
                values = [self.onehot_colnames[dummy_idx[i]-len(self.num_cols)].replace(c + '_', '') for i in max_col]
            dummies_df[c] = values  # TODO eliminate dummies_df use res in

        res = pd.DataFrame(X[:, 0:len(self.num_cols)], columns=self.num_cols)
        for c in dummies:
            res[c] = dummies_df[c]
        # scale back
        for i in range(len(self.num_cols)):
            c = self.num_cols[i]
            res[c] = int(res[c] * self.sd[i] + self.mean[i])
        # print(res.shape)
        # print(res.columns)
        reorder_res = pd.DataFrame()
        for c in self.ordered_columns:
            reorder_res[c] = res[c]
        print(reorder_res.shape)
        return reorder_res
