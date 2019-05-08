from allennlp.common.registrable import Registrable


class Callback(Registrable):
    def call(self, trainer: 'EventBasedTrainer') -> None:
        raise NotImplementedError
