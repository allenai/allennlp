from allennlp.interpret.influence_interpreters import SimpleInfluence
from allennlp_models.pretrained import load_predictor

from allennlp_models.pair_classification.predictors.textual_entailment import (
    TextualEntailmentPredictor,
)
from allennlp.models.archival import load_archive

roberta_nli_predictor = load_predictor("pair-classification-roberta-mnli")

archive_file = "../allennlp-models/mnli_bert_output/model.tar.gz"
train_file = "../allennlp-models/data/multinli_1.0_train_head.jsonl"
test_file = "../allennlp-models/data/multinli_1.0_dev_mismatched_head.jsonl"

# archive = load_archive(archive_file)
# model = archive.model
dataset_reader = roberta_nli_predictor._dataset_reader
# predictor = TextualEntailmentPredictor(model, dataset_reader)
simple_if = SimpleInfluence(
    roberta_nli_predictor,
    train_file,
    test_file,
    dataset_reader,
    params_to_freeze=["_text_field_embedder.token_embedder_tokens.transformer_model"],
    recur_depth=10,
)
simple_if.calculate_inflence_and_save("simple_if_output.jsonl")
