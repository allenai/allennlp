import os
import sys

import argparse

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from allennlp.common.tqdm import Tqdm
from allennlp.common import Params
from allennlp.models.archival import load_archive
from allennlp.data.iterators import BasicIterator
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file
from allennlp.nn.util import move_to_device

def main(serialization_directory: int,
         device: int,
         data: str,
         prefix: str,
         domain: str = None):
    """
    serialization_directory : str, required.
        The directory containing the serialized weights.
    device: int, default = -1
        The device to run the evaluation on.
    data: str, default = None
        The data to evaluate on. By default, we use the validation data from
        the original experiment.
    prefix: str, default=""
        The prefix to prepend to the generated gold and prediction files, to distinguish
        different models/data.
    domain: str, optional (default = None)
        If passed, filters the ontonotes evaluation/test dataset to only contain the
        specified domain. This overwrites the domain in the config file from the model,
        to allow evaluation on domains other than the one the model was trained on.
    """
    config = Params.from_file(os.path.join(serialization_directory, "config.json"))

    if domain is not None:
        # Hack to allow evaluation on different domains than the
        # model was trained on.
        config["dataset_reader"]["domain_identifier"] = domain
        prefix = f"{domain}_{prefix}"
    else:
        config["dataset_reader"].pop("domain_identifier", None)

    dataset_reader = DatasetReader.from_params(config['dataset_reader'])
    evaluation_data_path = data if data else config['validation_data_path']

    archive = load_archive(os.path.join(serialization_directory, "model.tar.gz"), cuda_device=device)
    model = archive.model
    model.eval()

    prediction_file_path = os.path.join(serialization_directory, prefix + "_predictions.txt")
    gold_file_path = os.path.join(serialization_directory, prefix + "_gold.txt")
    prediction_file = open(prediction_file_path, "w+")
    gold_file = open(gold_file_path, "w+")

    # Load the evaluation data and index it.
    print("reading evaluation data from {}".format(evaluation_data_path))
    instances = dataset_reader.read(evaluation_data_path)

    with torch.autograd.no_grad():
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(model.vocab)

        model_predictions = []
        batches = iterator(instances, num_epochs=1, shuffle=False)
        for batch in Tqdm.tqdm(batches):
            batch = move_to_device(batch, device)
            result = model(**batch)
            predictions = model.decode(result)
            model_predictions.extend(predictions["tags"])

        for instance, prediction in zip(instances, model_predictions):
            fields = instance.fields
            verb_index = fields["metadata"]["verb_index"]
            gold_tags = fields["metadata"]["gold_tags"]
            sentence = fields["metadata"]["words"]
            write_to_conll_eval_file(prediction_file, gold_file,
                                     verb_index, sentence, prediction, gold_tags)
        prediction_file.close()
        gold_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="write conll format srl predictions"
                                                 " to file from a pretrained model.")
    parser.add_argument('--path', type=str, help='the serialization directory.')
    parser.add_argument('--device', type=int, default=-1, help='the device to load the model onto.')
    parser.add_argument('--data', type=str, default=None, help='A directory containing a dataset to evaluate on.')
    parser.add_argument('--prefix', type=str, default="", help='A prefix to distinguish model outputs.')
    parser.add_argument('--domain', type=str, default=None, help='An optional domain to filter by for producing results.')
    args = parser.parse_args()
    main(args.path, args.device, args.data, args.prefix, args.domain)
