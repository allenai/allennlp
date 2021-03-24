from allennlp.predictors import TextClassifierPredictor
from allennlp.models.model import Model
import torch


class FakeModelForTestingInterpret(Model):
    def __init__(self, vocab, max_tokens=7, num_labels=2):
        super().__init__(vocab)
        self._max_tokens = max_tokens
        self.embedder = torch.nn.Embedding(vocab.get_vocab_size(), 16)
        self.linear = torch.nn.Linear(max_tokens * 16, num_labels)
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        tokens = tokens["tokens"]["tokens"][:, 0 : self._max_tokens]
        embedded = self.embedder(tokens)
        logits = self.linear(torch.flatten(embedded).unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}
        if label is not None:
            output_dict["loss"] = self._loss(logits, label.long().view(-1))
        return output_dict

    def make_output_human_readable(self, output_dict):
        preds = output_dict["probs"]
        if len(preds.shape) == 1:
            output_dict["probs"] = preds.unsqueeze(0)
            output_dict["logits"] = output_dict["logits"].unsqueeze(0)

        classes = []
        for prediction in output_dict["probs"]:
            label_idx = prediction.argmax(dim=-1).item()
            output_dict["loss"] = self._loss(output_dict["logits"], torch.LongTensor([label_idx]))
            label_str = str(label_idx)
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict


class FakePredictorForTestingInterpret(TextClassifierPredictor):
    def get_interpretable_layer(self):
        return self._model.embedder

    def get_interpretable_text_field_embedder(self):
        return self._model.embedder
