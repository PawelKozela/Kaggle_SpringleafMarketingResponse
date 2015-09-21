import pandas as pd
import numpy as np
from sklearn import preprocessing

from feature_engineering.feature_utilities import identify_feature, transform_date


NA_VALUE = -1


class SpringleafDataset:

    def __init__(self, train_file, test_file, sample_rate=1):
        self._train_file = train_file
        self._test_file = test_file

        self._train_features = None
        self._train_target = None
        self._test_features = None
        self._sample_rate = sample_rate

        self._feature_types = None

        self.load_train()
        if self._test_file is not None:
            self.load_test()

    def load_train(self):
        train_data = pd.read_hdf(self._train_file, 'data')[::self._sample_rate]

        self._train_target = pd.DataFrame(train_data['target'], index=train_data.index)
        self._train_features = train_data.drop('target', axis=1)

    def load_test(self):
        self._test_features = pd.read_hdf(self._test_file, 'data')

    def identify_feature_types(self, train_only=True):
        feature_types = []
        for i in range(0, len(self._train_features.columns)):
            if train_only:
                feature_types.append(identify_feature(self._train_features.iloc[:, i]))
            else:
                feature_types.append(self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]))

        self._feature_types = pd.DataFrame(feature_types)
        return self._feature_types

    def transform_object_features(self, train_only=True):
        feature_types = self.identify_feature_types(train_only)

        for i in range(0, len(feature_types)):
            if feature_types.iloc[i, 0] == 'Object' and feature_types.iloc[i, 1] == 'Date':
                self._train_features.iloc[:, i] = self._train_features.iloc[:, i].apply(transform_date)

                if not train_only:
                    self._test_features.iloc[:, i] = self._test_features.iloc[:, i].apply(transform_date)

            elif feature_types.iloc[i, 0] == 'Object':
                le = preprocessing.LabelEncoder()

                if train_only:
                    le.fit(self._train_features.iloc[:, i])
                else:
                    le.fit(self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]))

                self._train_features.iloc[:, i] = le.transform(self._train_features.iloc[:, i])

                if not train_only:
                    self._test_features.iloc[:, i] = le.transform(self._test_features.iloc[:, i])

    def transform_features_standard(self, train_only=True):

        self.transform_object_features(train_only)

        self._train_features['MOD_0217'] = np.mod(self._train_features['VAR_0217'], 7)
        self._train_features['MOD_0075'] = np.mod(self._train_features['VAR_0075'], 7)
        self._train_features.fillna(NA_VALUE, inplace=True)

        if not train_only:
            self._test_features['MOD_0217'] = np.mod(self._test_features['VAR_0217'], 7)
            self._test_features['MOD_0075'] = np.mod(self._test_features['VAR_0075'], 7)
            self._test_features.fillna(NA_VALUE, inplace=True)
