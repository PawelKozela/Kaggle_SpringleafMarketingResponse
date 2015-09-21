import os
import pandas as pd


class BaseFeatureTransformer:

    def __init__(self, features_dir):
        self.features_dir = features_dir

        # Load features
        self._train_features = pd.read_hdf(os.path.join(features_dir, 'train_features.hf5'), 'data')
        self._test_features = pd.read_hdf(os.path.join(features_dir, 'test_features.hf5'), 'data')

    def transform_features(self):
        raise NotImplementedError

    def save_features(self, target_dir):
        # To avoid errors
        if self.features_dir == target_dir:
            raise ValueError('Would override base features')

        self._train_features.to_hdf(os.path.join(target_dir, 'train_features.hf5'), 'data', complib='blosc', complevel=9)
        self._test_features.to_hdf(os.path.join(target_dir, 'test_features.hf5'), 'data', complib='blosc', complevel=9)