import os
import time

import h5py
import numpy as np
import pandas as pd

from base.base_batch_generator import BaseBatchGenerator
from utils.config import process_config
from utils.logging import logger


class Dataset(BaseBatchGenerator):
    def __init__(self, config, subset='test', shuffle=False):
        super().__init__(config)
        self.batch_size = config.data_loader.batch_size[subset] * config.model.num_gpus
        self.class_weight = {int(k): v for k, v in self.config.trainer.class_weight.items()}
        self.df = pd.read_csv(config.data_loader.df_file, index_col=0)
        self.df = self.df.loc[self.df.Fold1 == subset].reset_index(drop=True)
        self.data = None
        self.labels = None
        self.indexes = None
        self.num_subjects = len(self.df)
        self.shuffle = shuffle
        self.subset = subset
        self.load_data()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(sum(self.df.Length.values) / self.batch_size))

    def __getitem__(self, idx):
        current_batch_indices = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        data, labels, weights = self.__data_generation(current_batch_indices)
        return data, labels, weights

    def on_epoch_end(self):
        self.indexes = [(i, j) for i in np.arange(self.num_subjects) for j in np.arange(self.df.Length[i])]
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_files):

        data = [[]] * self.batch_size
        labels = [[]] * self.batch_size

        # Generate data
        for i, (subject_id, chunk_id) in enumerate(batch_files):
            data[i] = self.data[subject_id][chunk_id, :, :, :]
            labels[i] = self.labels[subject_id][chunk_id, :, :]

        data = np.stack([datum for datum in data if not datum == []], axis=0)
        labels = np.stack([label for label in labels if not label == []], axis=0).astype(int)
        weights = np.squeeze(labels)
        weights = np.vectorize(self.class_weight.__getitem__)(weights)

        return data, labels, weights

    def __load_h5data(self, file_id, chunk_id=None):

        with h5py.File(os.path.join(self.config.data_loader.data_dir, file_id + '.h5'), 'r') as f:
            if not chunk_id:
                data = np.expand_dims(f['data'][:], axis=-1)
                labels = f['labels'][:]
            else:
                data = np.expand_dims(f['data'][chunk_id, :, :], axis=-1)
                labels = f['labels'][chunk_id, :]

        if self.config.data_loader.data_format == 'channels_first':
            data = np.transpose(data, [2, 0, 1])

        return data, labels

    def load_data(self):
        self.data = [[]] * self.num_subjects
        self.labels = [[]] * self.num_subjects
        for subject_idx in range(self.num_subjects):
            if subject_idx % 100 == 1:
                logger('Loaded {} out of {} subjects |'.format(subject_idx, self.num_subjects))
            file_id = self.df.FileID.iloc[subject_idx].lower()
            self.data[subject_idx], self.labels[subject_idx] = self.__load_h5data(file_id)
        # self.data = np.stack([datum for datum in self.data if not datum == []], axis=0)
        # self.labels = np.stack([label for label in self.labels if not label == []], axis=0).astype(int)


class TrainDataset(Dataset):
    def __init__(self, config):
        super().__init__(config, subset='train', shuffle=True)


class EvalDataset(Dataset):
    def __init__(self, config):
        super().__init__(config, subset='eval')


class TestDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)


if __name__ == '__main__':
    config = process_config('./configs/test.json')
    train_data = TrainDataset(config)

    start = time.time()
    for idx, (x, y, w) in enumerate(train_data):
        pass
        # print('{} | X shape: {} | Y shape: {}'.format(idx, x.shape, y.shape))
    end = time.time()

    print('Elapsed time: {} '.format(end - start))
    # eval_generator = EvalGenerator(config)
    # test_generator = TestGenerator(config)
    print('Hej')
