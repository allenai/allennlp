import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

import argparse
import torch
import tqdm
from allennlp.common import Params
from allennlp.data.iterators import BasicIterator
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.nn.util import arrays_to_variables


def map_output_to_instances(output_dict, instances):
    output_list = []
    for i, instance in enumerate(instances):
        instance_dict = {
                "tokens": instance.fields["tokens"].tokens,
                "gold_tags": instance.fields["tags"].labels
        }

        instance_dict["per_element_loss"] = output_dict["per_element_loss"][i]
        instance_dict["sequence_loss"] = output_dict["sequence_loss"][i]
        instance_dict["tags"] = output_dict["tags"][i]
        instance_dict["viterbi_score"] = output_dict["scores"][i]

        output_list.append(instance_dict)

    return output_list


def sanitise_outputs(output_dict):
    for name, output in list(output_dict.items()):
        output = output[0]
        if isinstance(output, torch.autograd.Variable):
            output = output.data.cpu().numpy().tolist()
        output_dict[name] = output
    return output_dict


def main(serialization_directory: str, device: int):
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

    print("Reading evaluation data from {}".format(evaluation_data_path))
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(model._vocab)
    iterator = BasicIterator(batch_size=32)

    all_results = []
    index = 0
    for batch in tqdm.tqdm(iterator(dataset, num_epochs=1, shuffle=False)):
        try:
            raw_instances = dataset.instances[32 * index: 32 * (index + 1)]
        # last batch might not be full
        except IndexError:
            raw_instances = dataset.instances[32 * index:]

        tensor_batch = arrays_to_variables(batch, device, for_training=False)
        result = model.decode(model.forward(**tensor_batch))
        separated_outputs = map_output_to_instances(sanitise_outputs(result), raw_instances)
        all_results.extend(separated_outputs)

        index += 1

    with open("validation_stats.json", "w") as out_file:
        json.dump(all_results, out_file, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate per instance statistics NLL on the validation"
                                                 "set for a SRL model.")

    parser.add_argument('--path', type=str, help='The serialization directory.')
    parser.add_argument('--device', type=int, default=-1, help='The device to load the model onto.')

    args = parser.parse_args()
    main(args.path, args.device)
