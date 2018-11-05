import os
import time

import h5py
import numpy as np
import pandas as pd

from base.base_batch_generator import BaseBatchGenerator
from utils.config import process_config


class DataGenerator(BaseBatchGenerator):
    def __init__(self, config, subset='test', shuffle=False):
        super().__init__(config)
        self.batch_size = config.data_loader.batch_size[subset] * config.model.num_gpus
        self.df = pd.read_csv(config.data_loader.df_file, index_col=0)
        self.indexes = None
        self.num_subjects = len(self.df)
        self.shuffle = shuffle
        self.subset = subset
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(sum(self.df.Length.values) / self.batch_size))

    def __getitem__(self, idx):
        current_batch_indices = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        data, labels = self.__data_generation(current_batch_indices)
        return data, labels

    def on_epoch_end(self):
        self.indexes = [(i, j) for i in np.arange(self.num_subjects) for j in np.arange(self.df.Length[i])]
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_files):

        data = [[]] * self.batch_size
        labels = [[]] * self.batch_size

        # Generate data
        for i, (subject_id, chunk_id) in enumerate(batch_files):
            file_id = self.df.FileID.iloc[subject_id].lower()
            data[i], labels[i] = self.__load_h5data(file_id, chunk_id)

        data = np.stack([datum for datum in data if not datum == []], axis=0)
        labels = np.stack([label for label in labels if not label == []], axis=0).astype(int)

        return data, labels

    def __load_h5data(self, file_id, chunk_id):

        with h5py.File(os.path.join(self.config.data_loader.data_dir, file_id + '.h5'), 'r') as f:
            data = np.expand_dims(f['data'][chunk_id, :, :], axis=2)
            labels = f['labels'][chunk_id, :]

        if self.config.data_loader.data_format == 'channels_first':
            data = np.transpose(data, [2, 0, 1])

        return data, labels


class TrainGenerator(DataGenerator):
    def __init__(self, config):
        super().__init__(config, subset='train', shuffle=True)


class EvalGenerator(DataGenerator):
    def __init__(self, config):
        super().__init__(config, subset='eval')


class TestGenerator(DataGenerator):
    def __init__(self, config):
        super().__init__(config)


if __name__ == '__main__':
    config = process_config('./configs/test.json')
    train_generator = TrainGenerator(config)

    start = time.time()
    for idx, (x, y) in enumerate(train_generator):
        # pass
        print('{} | X shape: {} | Y shape: {}'.format(idx, x.shape, y.shape))
    end = time.time()

    print('Elapsed time: {} '.format(end - start))
    # eval_generator = EvalGenerator(config)
    # test_generator = TestGenerator(config)
    print('Hej')
