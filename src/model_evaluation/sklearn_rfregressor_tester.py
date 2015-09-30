from base_tester import BaseTester
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score


class RFRegressorTester(BaseTester):

    RANGE_NB_TREES = [10, 50, 100]
    RANGE_MAX_DEPTH = [6, 12]
    RANGE_MAX_FEATURES = [.1, .5, 1.]
    RANGE_MIN_SAMPLES_LEAF = [1, 10, 100]

    def evaluate_model(self, features_model, x_train, y_train, x_validation, y_validation):

        for i_nb_trees in self.RANGE_NB_TREES:
            for i_max_depth in self.RANGE_MAX_DEPTH:
                for i_max_features in self.RANGE_MAX_FEATURES:
                    for i_min_samples_leaf in self.RANGE_MIN_SAMPLES_LEAF:

                        rf = RandomForestRegressor(n_estimators=i_nb_trees, max_depth=i_max_depth, max_features=i_max_features, min_samples_leaf=i_min_samples_leaf, n_jobs=2)
                        rf.fit(x_train, y_train)
                        score = roc_auc_score(y_validation, rf.predict(x_validation))

                        timestamp = datetime.now()
                        model_id = 'RFRegressor_{:%Y%m%d_%H%M%S}'.format(timestamp)
                        result = {'model_id': model_id,
                                  'model_type': 'RFRegressor',
                                  'timestamp': timestamp,
                                  'features_model': features_model,
                                  'score': score,
                                  'nb_trees': i_nb_trees,
                                  'max_depth': i_max_depth,
                                  'max_features': i_max_features,
                                  'min_samples_leaf': i_min_samples_leaf}

                        print result

                        self.save_result(result)