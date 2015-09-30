import os
import pickle


class BaseFeatureSetValidator:
    """
    The intent of this class is to compare different pre-processing strategies
    Each subclass always evaluate with the same model and stores the results
    """

    def __init__(self, data_dir, results_dir):
        self._data_dir = data_dir
        self._results_dir = results_dir
        self._results_filename = os.path.join(self._results_dir, 'features_results.pickle')

    def _save_results(self, result):
        if os.path.exists(self._results_filename):
            results = pickle.load(open(self._results_filename, 'rb'))
        else:
            results = []

        results.append(result)

        pickle.dump(results, open(self._results_filename, 'wb'))

    def evaluate_feature_set(self, feature_set_name):
        raise NotImplementedError