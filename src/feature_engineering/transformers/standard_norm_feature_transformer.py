from sklearn import preprocessing
import numpy as np

from base_feature_transformer import BaseFeatureTransformer
from .. import feature_utilities


NA_VALUE = -1


class StandardNormFeatureTransformer(BaseFeatureTransformer):

    def transform_object_features(self):
        for i in range(0, self._train_features.shape[1]):
            feature_type = feature_utilities.identify_feature(self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]))

            if feature_type[0] == 'Object' and feature_type[1] == 'Date':
                self._train_features.iloc[:, i] = self._train_features.iloc[:, i].apply(feature_utilities.transform_date)
                self._test_features.iloc[:, i] = self._test_features.iloc[:, i].apply(feature_utilities.transform_date)

            elif feature_type[0] == 'Object':
                le = preprocessing.LabelEncoder()
                le.fit(self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]))

                self._train_features.iloc[:, i] = le.transform(self._train_features.iloc[:, i])
                self._test_features.iloc[:, i] = le.transform(self._test_features.iloc[:, i])

    def transform_features(self):

        self.transform_object_features()

        self._train_features['MOD_0217'] = np.mod(self._train_features['VAR_0217'], 7)
        self._train_features['MOD_0075'] = np.mod(self._train_features['VAR_0075'], 7)
        self._train_features.fillna(NA_VALUE, inplace=True)

        self._test_features['MOD_0217'] = np.mod(self._test_features['VAR_0217'], 7)
        self._test_features['MOD_0075'] = np.mod(self._test_features['VAR_0075'], 7)
        self._test_features.fillna(NA_VALUE, inplace=True)

        self.normalize_features()

    def normalize_features(self):
        self._train_features = self._train_features.astype(float)
        self._test_features = self._test_features.astype(float)

        for i in range(0, self._train_features.shape[1]):
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]))

            self._train_features.iloc[:, i] = scaler.transform(self._train_features.iloc[:, i])
            self._test_features.iloc[:, i] = scaler.transform(self._test_features.iloc[:, i])
