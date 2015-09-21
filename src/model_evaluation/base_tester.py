import os, pickle


class BaseTester:

    def __init__(self, results_dir):
        self._results_dir = results_dir
        self._results_filename = os.path.join(self._results_dir, 'results.pickle')

    def evaluate_model(self, features_model, x_train, y_train, x_validation, y_validation):
        raise NotImplementedError

    def save_result(self, result):
        if os.path.exists(self._results_filename):
            results = pickle.load(open(self._results_filename, 'rb'))
        else:
            results = []

        results.append(result)

        pickle.dump(results, open(self._results_filename, 'wb'))