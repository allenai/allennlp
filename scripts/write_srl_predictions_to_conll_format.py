import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
import argparse
from allennlp.common import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data.iterators import BasicIterator
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file


def main(serialization_directory, device):
    """
    serialization_directory : str, required.
        The directory containing the serialized weights.
    device: int, default = -1
        The device to run the evaluation on.
    """

    config = Params.from_file(os.path.join(serialization_directory, "model_params.json"))
    dataset_reader = DatasetReader.from_params(config['dataset_reader'])
    evaluation_data_path = config['validation_data_path']

    model = Model.load(config, serialization_dir=serialization_directory, cuda_device=device)

    prediction_file_path = os.path.join(serialization_directory, "predictions.txt")
    gold_file_path = os.path.join(serialization_directory, "gold.txt")
    prediction_file = open(prediction_file_path, "w+")
    gold_file = open(gold_file_path, "w+")

    # Load the evaluation data and index it.
    print("Reading evaluation data from {}".format(evaluation_data_path))
    instances = dataset_reader.read(evaluation_data_path)
    iterator = BasicIterator(batch_size=32)
    iterator.index_with(model.vocab)

    model_predictions = []
    batches = iterator(instances, num_epochs=1, shuffle=False, cuda_device=device, for_training=False)
    for batch in Tqdm.tqdm(batches):
        result = model(**batch)
        predictions = model.decode(result)
        model_predictions.extend(predictions["tags"])

    for instance, prediction in zip(instances, model_predictions):
        fields = instance.fields
        try:
            # Most sentences have a verbal predicate, but not all.
            verb_index = fields["verb_indicator"].labels.index(1)
        except ValueError:
            verb_index = None

        gold_tags = fields["tags"].labels
        sentence = fields["tokens"].tokens

        write_to_conll_eval_file(prediction_file, gold_file,
                                 verb_index, sentence, prediction, gold_tags)
    prediction_file.close()
    gold_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Write CONLL format SRL predictions"
                                                 " to file from a pretrained model.")
    parser.add_argument('--path', type=str, help='The serialization directory.')
    parser.add_argument('--device', type=int, default=-1, help='The device to load the model onto.')

    args = parser.parse_args()
    main(args.path, args.device)
