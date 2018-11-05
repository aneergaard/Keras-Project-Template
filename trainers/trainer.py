import os

from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from base.base_trainer_generator import BaseTrainGenerator
from data_loader.data_loader import TrainGenerator, EvalGenerator
from models.model import CustomModel
from utils.config import process_config


class Trainer(BaseTrainGenerator):
    def __init__(self, model, data, config):
        super().__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.schedule_fn = self.step_decay
        self.init_callbacks()

    def step_decay(self, epoch, learning_rate):
        lr = self.config.model.initial_lr * (self.config.callbacks.learningrate_decay_rate ** (
                    epoch // self.config.callbacks.learningrate_decay_epochs))
        return lr

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.callbacks.earlystopping_monitor,
                patience=self.config.callbacks.earlystopping_patience,
                verbose=self.config.callbacks.earlystopping_verbose,
            )
        )

        self.callbacks.append(
            LearningRateScheduler(self.schedule_fn, verbose=1)
        )

    def train(self):
        history = self.model.fit_generator(
            self.train_data,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks,
            validation_data=self.eval_data,
            workers=self.config.trainer.num_workers,
            use_multiprocessing=self.config.trainer.use_multiprocessing,
            max_queue_size=self.config.trainer.max_queue_size
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])


if __name__ == '__main__':

    config = process_config('./configs/test.json')
    data = {'train': TrainGenerator(config),
            'eval': EvalGenerator(config)}
    model = CustomModel(config)

    trainer = Trainer(model, data, config)
    print(trainer)
