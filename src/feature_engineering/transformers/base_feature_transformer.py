import os
import pandas as pd


class BaseFeatureTransformer:

    def __init__(self, features_dir, train_features_filename='train_features.hf5', test_features_filename='test_features.hf5', train_y_filename='train_train_y.hf5'):
        self.features_dir = features_dir
        self._train_features_filename = train_features_filename
        self._test_features_filename = test_features_filename
        self._train_y_filename = train_y_filename

        # Load features
        self._train_features = pd.read_hdf(os.path.join(features_dir, train_features_filename), 'data')
        self._test_features = pd.read_hdf(os.path.join(features_dir, test_features_filename), 'data')
        self._train_y = pd.read_hdf(os.path.join(features_dir, train_y_filename), 'data')

    def transform_features(self):
        raise NotImplementedError

    def save_features(self, target_dir):
        # To avoid errors
        if self.features_dir == target_dir:
            raise ValueError('Would override base features')

        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        self._train_features.to_hdf(os.path.join(target_dir, self._train_features_filename), 'data', complib='blosc', complevel=9)
        self._test_features.to_hdf(os.path.join(target_dir, self._test_features_filename), 'data', complib='blosc', complevel=9)