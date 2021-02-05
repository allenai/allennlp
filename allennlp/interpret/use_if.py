from allennlp_models.pair_classification.dataset_readers.snli import CollapsedSnliReader
from allennlp.interpret.influence_interpreters import SimpleInfluence
from allennlp_models.pair_classification.predictors.textual_entailment import (
    TextualEntailmentPredictor,
)
from allennlp.models.archival import load_archive

archive_file = "../allennlp-models/mnli_bert_output/model.tar.gz"
train_file = "../allennlp-models/data/multinli_1.0_train_head.jsonl"
test_file = "../allennlp-models/data/multinli_1.0_dev_mismatched_head.jsonl"

archive = load_archive(archive_file)
model = archive.model
dataset_reader = archive.validation_dataset_reader
predictor = TextualEntailmentPredictor(model, dataset_reader)
simple_if = SimpleInfluence(predictor, train_file, test_file, dataset_reader)
simple_if.calculate_inflence_and_save("simple_if_output.jsonl")
