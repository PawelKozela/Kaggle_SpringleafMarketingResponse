from datetime import datetime
import os

import pandas as pd
import xgboost as xgb

from base_feature_set_validator import BaseFeatureSetValidator


class FastFeatureSetValidator(BaseFeatureSetValidator):
    """
    Evaluates the feature sets with a short XGBoost: 100 rounds, 3 seeds
    Takes ~ 3 minutes
    """

    _max_depth = 6
    _eta = .1
    _col_sample = .1
    _sub_sample = 1
    _nb_rounds = 100

    NB_SEEDS = 3

    def _evaluate_data(self, feature_set_name):
        x_train = pd.read_hdf(os.path.join(self._data_dir, feature_set_name, 'train_train_features.hf5'), 'data')
        y_train = pd.read_hdf(os.path.join(self._data_dir, 'train_train_y.hf5'), 'data')

        x_validation = pd.read_hdf(os.path.join(self._data_dir, feature_set_name, 'train_validation_features.hf5'), 'data')
        y_validation = pd.read_hdf(os.path.join(self._data_dir, 'train_validation_y.hf5'), 'data')

        dtrain = xgb.DMatrix(x_train, y_train['target'], missing=-1)
        dtest = xgb.DMatrix(x_validation, y_validation['target'], missing=-1)

        results = {}
        for i_seed in range(0, self.NB_SEEDS):
            evals_result = {}
            params = {'bst:max_depth': self._max_depth,
                      'bst:eta': self._eta,
                      'objective': 'binary:logistic',
                      'colsample_bytree': self._col_sample,
                      'subsample': self._sub_sample,
                      'eval_metric': 'auc',
                      'silent': 1,
                      'nthread': 16,
                      'seed': i_seed}

            eval_list = [(dtest, 'eval')]
            bst = xgb.train(params, dtrain, self._nb_rounds, eval_list, evals_result=evals_result, verbose_eval=False)
            results[i_seed] = evals_result['eval'][-1]

        return results

    def evaluate_feature_set(self, feature_set_name):
        feature_set_results = self._evaluate_data(feature_set_name)

        result = {
            'feature_set': feature_set_name,
            'validator': 'FastFeatureValidator',
            'timestmap': datetime.now(),
            'results': feature_set_results
        }

        print result

        self._save_results(result)