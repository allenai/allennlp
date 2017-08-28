import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
import argparse
from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file


def main(serialization_directory, device):
    """

    config_file : str, required.
        A config file containing a model specification.
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

    # Load the evaluation data
    print("Reading evaluation data from {}".format(evaluation_data_path))
    dataset = dataset_reader.read(evaluation_data_path)

    # We have to do this using the .tag() method, rather than in bulk, because we need
    # to use the constrained inference to guarantee valid tag sequences.
    for instance in dataset.instances:
        fields = instance.fields
        results = model.tag(fields["tokens"], fields["verb_indicator"])

        try:
            # Most sentences have a verbal predicate, but not all.
            verb_index = fields["verb_indicator"].labels.index(1)
        except ValueError:
            verb_index = None

        gold_tags = fields["tags"].labels
        predicted_tags = results["tags"]
        sentence = fields["tokens"].tokens

        write_to_conll_eval_file(prediction_file, gold_file,
                                 verb_index, sentence, gold_tags, predicted_tags)
    prediction_file.close()
    gold_file.close()

    # Run the perl evaluation script on the written files.
    perl_script_command = ["perl", "scripts/srl-eval.pl", prediction_file_path, gold_file_path]
    subprocess.call(perl_script_command, stdout=os.path.join(serialization_directory, "results.txt"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Write CONLL format SRL predictions"
                                                 " to file from a pretrained model.")
    parser.add_argument('--path', type=str, help='The serialization directory.')
    parser.add_argument('--device', type=int, default=-1, help='The device to load the model onto.')

    args = parser.parse_args()
    main(args.path, args.device)
