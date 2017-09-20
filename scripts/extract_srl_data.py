import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
import tqdm
from allennlp.common import Params
from allennlp.data.iterators import BasicIterator
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.nn.util import arrays_to_variables


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

    print("Reading evaluation data from {}".format(evaluation_data_path))
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(model._vocab)
    iterator = BasicIterator(batch_size=32)

    all_results = []
    for batch in tqdm.tqdm(iterator(dataset, num_epochs=1, shuffle=False)):
        tensor_batch = arrays_to_variables(batch, device, for_training=False)
        result = model.decode(model.forward(**tensor_batch))
        all_results.append(result)

        print(result)

        break



