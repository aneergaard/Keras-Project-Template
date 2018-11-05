import sys
from datetime import datetime

import keras.backend as k
import tensorflow as tf

from utils import factory
from utils.args import get_args
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logging import logger


def main():
    ##########################################################
    # TensorFlow configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    k.tensorflow_backend.set_session(tf.Session(config=tf_config))
    ##########################################################

    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)

        # create the experiments dirs
        create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

        logger('Creating data generators ...'.format(datetime.now()))
        data_loader = {'train': factory.create("data_loader."+config.data_loader.name)(config, subset='train', shuffle=True),
                       'eval': factory.create("data_loader." + config.data_loader.name)(config, subset='eval')}

        logger('Creating the model ...'.format(datetime.now()))
        model = factory.create("models."+config.model.name)(config)

        logger('Creating the trainer ...'.format(datetime.now()))
        if config.model.num_gpus > 1:
            trainer = factory.create("trainers."+config.trainer.name)(model.parallel_model, data_loader, config)
        else:
            trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader, config)

        logger('Starting model training ...'.format(datetime.now()))
        trainer.train()
        
        logger('Training has finished!'.format(datetime.now()))

    except Exception as e:
        logger(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
