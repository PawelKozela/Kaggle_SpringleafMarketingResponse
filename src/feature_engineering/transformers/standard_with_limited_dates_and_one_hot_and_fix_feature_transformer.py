from sklearn import preprocessing
import numpy as np
import pandas as pd

from base_feature_transformer import BaseFeatureTransformer
from .. import feature_utilities

NA_VALUE = -1


class StandardWithLimitedDatesAndOneHotAndFixFeatureTransformer(BaseFeatureTransformer):
    DATE_FEATURES = ['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204', 'VAR_0217']
    FULL_DATE_FEATURES = ['VAR_0073', 'VAR_0075', 'VAR_0217']

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
                print self._train_features.columns[i]
                # self._train_features.iloc[:, i] = 0
                # self._test_features.iloc[:, i] = 0
                var = self._train_features.columns[i]
                tmp = pd.DataFrame({var: self._train_features.iloc[:, i], 'target': self._train_y['target']})
                tmp['tmp'] = 1

                tmp_pivot = pd.pivot_table(tmp, columns='target', index=var, values='tmp', aggfunc='count')
                #tmp_pivot.iloc[:, 0].fillna(0)
                #tmp_pivot.iloc[:, 1].fillna(0)

                tmp_pivot['Ratio'] = tmp_pivot.iloc[:, 1] / (tmp_pivot.iloc[:, 0] + tmp_pivot.iloc[:, 1])
                tmp_pivot['Index'] = tmp_pivot.index
                #tmp_pivot.fillna(0.23)  # Just in case there are still missing values

                self._train_features.iloc[:, i] = self._train_features.iloc[:, i].apply(feature_utilities.replace_with_avg, args=(tmp_pivot,))
                self._test_features.iloc[:, i] = self._test_features.iloc[:, i].apply(feature_utilities.replace_with_avg, args=(tmp_pivot,))

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

    def add_date_relations(self):
        for i in range(0, len(self.FULL_DATE_FEATURES)):
            for j in range(0, len(self.FULL_DATE_FEATURES)):
                if i == j:
                    continue
                feature_name = '{}_{}'.format(self.FULL_DATE_FEATURES[i], self.FULL_DATE_FEATURES[j])
                self._train_features[feature_name] = self._train_features[self.FULL_DATE_FEATURES[i]] - self._train_features[self.FULL_DATE_FEATURES[j]]
                self._test_features[feature_name] = self._test_features[self.FULL_DATE_FEATURES[i]] - self._test_features[self.FULL_DATE_FEATURES[j]]

    def transform_features(self):

        raw_train_features = self._train_features.copy()
        raw_test_features = self._test_features.copy()

        self.transform_object_features()

        for feature_name in self.FULL_DATE_FEATURES:
            # Day of week
            self._train_features[feature_name.replace('VAR', 'DOW')] = raw_train_features[feature_name].apply(feature_utilities.get_day_of_week)
            self._test_features[feature_name.replace('VAR', 'DOW')] = raw_test_features[feature_name].apply(feature_utilities.get_day_of_week)

            # Day of month
            self._train_features[feature_name.replace('VAR', 'DOM')] = raw_train_features[feature_name].apply(feature_utilities.get_day_of_month)
            self._test_features[feature_name.replace('VAR', 'DOM')] = raw_test_features[feature_name].apply(feature_utilities.get_day_of_month)

            # Day of year
            self._train_features[feature_name.replace('VAR', 'DOY')] = raw_train_features[feature_name].apply(feature_utilities.get_day_of_year)
            self._test_features[feature_name.replace('VAR', 'DOY')] = raw_test_features[feature_name].apply(feature_utilities.get_day_of_year)

            # Month
            self._train_features[feature_name.replace('VAR', 'MON')] = raw_train_features[feature_name].apply(feature_utilities.get_month)
            self._test_features[feature_name.replace('VAR', 'MON')] = raw_test_features[feature_name].apply(feature_utilities.get_month)

        # VAR_0204 parsing
        self._train_features['VAR_0204_H'] = raw_train_features['VAR_0204'].apply(feature_utilities.get_var_204_value)
        self._test_features['VAR_0204_H'] = raw_test_features['VAR_0204'].apply(feature_utilities.get_var_204_value)

        self.add_date_relations()

        # ???
        self._train_features.fillna(NA_VALUE, inplace=True)
        self._test_features.fillna(NA_VALUE, inplace=True)

