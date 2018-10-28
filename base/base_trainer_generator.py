class BaseTrainGenerator(object):
    def __init__(self, model, data, config):
        self.model = model
        self.train_data = data['train']
        self.eval_data = data['eval']
        self.config = config

    def train(self):
        raise NotImplementedError
