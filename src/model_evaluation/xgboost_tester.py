from base_tester import BaseTester
import xgboost as xgb
from datetime import datetime


class XGBoostTester(BaseTester):

    RANGE_ETA = [.005, .01, .05, .1]
    RANGE_MAX_DEPTH = [6, 12]
    RANGE_COL_SAMPLE = [.05, .1, .25, 1]
    RANGE_SUB_SAMPLE = [.5, 1]
    RANGE_MIN_CHILD_WEIGHT = [1, 10, 100]

    NB_ROUNDS = 5 * 1000

    def evaluate_model(self, features_model, x_train, y_train, x_validation, y_validation):
        dtrain = xgb.DMatrix(x_train, y_train['target'])
        dtest = xgb.DMatrix(x_validation, y_validation['target'])

        for i_eta in self.RANGE_ETA:
            for i_max_depth in self.RANGE_MAX_DEPTH:
                for i_col_sample in self.RANGE_COL_SAMPLE:
                    for i_sub_sample in self.RANGE_SUB_SAMPLE:
                        for i_min_child_weight in self.RANGE_MIN_CHILD_WEIGHT:
                            evals_result = {}
                            params = {'bst:max_depth': i_max_depth,
                                      'bst:eta': i_eta,
                                      'objective': 'binary:logistic',
                                      'colsample_bytree': i_col_sample,
                                      'subsample': i_sub_sample,
                                      'min_child_weight': i_min_child_weight,
                                      'eval_metric': 'auc',
                                      'silent': 1,
                                      'nthread': 4}

                            eval_list = [(dtrain, 'train'), (dtest, 'eval')]
                            bst = xgb.train(params, dtrain, self.NB_ROUNDS, eval_list, evals_result=evals_result, verbose_eval=False, early_stopping_rounds=200)

                            timestamp = datetime.now()
                            model_id = 'XGBoost_{:%Y%m%d_%H%M%S}'.format(timestamp)
                            result = {'model_id': model_id,
                                      'model_type': 'XGBoost',
                                      'timestamp': timestamp,
                                      'features_model': features_model,
                                      'score': evals_result['eval'][-1],
                                      'nb_rounds_max': self.NB_ROUNDS,
                                      'nb_rounds': len(evals_result['eval']),
                                      'params': params}

                            print result

                            self.save_result(result)