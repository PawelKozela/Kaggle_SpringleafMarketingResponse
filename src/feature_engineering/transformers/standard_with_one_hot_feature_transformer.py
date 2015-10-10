from sklearn import preprocessing
import numpy as np
import pandas as pd

from base_feature_transformer import BaseFeatureTransformer
from .. import feature_utilities

NA_VALUE = -1


class StandardWithOneHotFeatureTransformer(BaseFeatureTransformer):

    def transform_object_features(self):

        features_for_one_hot = []
        for i in range(0, self._train_features.shape[1]):
            feature_type = feature_utilities.identify_feature(self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]))
            feature_values = self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]).value_counts()

            if feature_type[0] == 'Object' and feature_type[1] == 'Date':
                self._train_features.iloc[:, i] = self._train_features.iloc[:, i].apply(feature_utilities.transform_date)
                self._test_features.iloc[:, i] = self._test_features.iloc[:, i].apply(feature_utilities.transform_date)

            elif feature_type[0] == 'Object' and 1 < feature_values.shape[0] < 60:
                features_for_one_hot.append(self._train_features.columns[i])

            elif feature_type[0] == 'Object':
                le = preprocessing.LabelEncoder()
                le.fit(self._train_features.iloc[:, i].append(self._test_features.iloc[:, i]))

                self._train_features.iloc[:, i] = le.transform(self._train_features.iloc[:, i])
                self._test_features.iloc[:, i] = le.transform(self._test_features.iloc[:, i])

        # One hot part
        train_features_encoded = pd.get_dummies(self._train_features, columns=features_for_one_hot)
        test_features_encoded = pd.get_dummies(self._test_features, columns=features_for_one_hot)

        train_columns = train_features_encoded.columns.tolist()
        test_columns = test_features_encoded.columns.tolist()

        for column in (set(train_columns) - set(test_columns)):
            test_features_encoded[column] = 0

        for column in (set(test_columns) - set(train_columns)):
            train_features_encoded[column] = 0

        # Align columns
        train_columns = train_features_encoded.columns.tolist()
        train_columns.sort()

        self._train_features = train_features_encoded[train_columns]
        self._test_features = test_features_encoded[train_columns]

    def transform_features(self):

        self.transform_object_features()

        # ???
        self._train_features.fillna(NA_VALUE, inplace=True)
        self._test_features.fillna(NA_VALUE, inplace=True)

