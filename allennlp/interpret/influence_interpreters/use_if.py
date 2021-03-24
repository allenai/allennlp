from allennlp.interpret.influence_interpreters import FastInfluence
from allennlp_models.pretrained import load_predictor
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder, EmptyEmbedder
from allennlp.common import Params
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.interpret.influence_interpreters.influence_utils import FAISSSnliWrapper

# from allennlp.data
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BertPooler
from copy import deepcopy
from allennlp_models.pair_classification.predictors.textual_entailment import (
    TextualEntailmentPredictor,
)
from allennlp_models.pair_classification import SnliReader
from allennlp.models.archival import load_archive

archive_file = "../../../../allennlp-models/mnli_bert_output/model.tar.gz"
train_file = "../../../../allennlp-models/data/multinli_1.0_train_first50.jsonl"
test_file = "../../../../allennlp-models/data/multinli_1.0_dev_mismatched_head.jsonl"

# archive = load_archive(archive_file)
# model = archive.model

# predictor = TextualEntailmentPredictor(model, dataset_reader)
# simple_if = SimpleInfluence(
#     roberta_nli_predictor,
#     train_file,
#     test_file,
#     dataset_reader,
#     params_to_freeze=["_text_field_embedder.token_embedder_tokens.transformer_model"],
#     recur_depth=10,
# )
transformer_name = "bert-base-uncased"
params = Params.from_file("use_if.jsonnet")
dataset_reader_params = params.pop("dataset_reader")
text_field_embedder_params = params.pop("text_field_embedder")
seq2vec_params = params.pop("seq2vec_encoder")

text_field_embedder = BasicTextFieldEmbedder.from_params(text_field_embedder_params)
faiss_dataset_reader = DatasetReader.from_params(dataset_reader_params)
seq2vec = Seq2VecEncoder.from_params(seq2vec_params)
faiss_vocab = Vocabulary.from_pretrained_transformer(transformer_name)

faiss_snli_wrapper = FAISSSnliWrapper(faiss_vocab, text_field_embedder, seq2vec)

# token_embedder = PretrainedTransformerEmbedder(transformer_name)
# text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedder, "label": EmptyEmbedder()})

roberta_nli_predictor = load_predictor("pair-classification-roberta-mnli")
dataset_reader = roberta_nli_predictor._dataset_reader

# faiss_dataset_reader = SnliReader(token_indexers={"tokens": token_indexer}, combine_input_fields=True)
# faiss_dataset_reader._tokenizer = tokenizer
# faiss_dataset_reader._token_indexers = {"tokens": token_indexer}

# simple_if.calculate_inflence_and_save("simple_if_output.jsonl")
fast_if = FastInfluence(
    roberta_nli_predictor,
    train_file,
    test_file,
    dataset_reader,
    faiss_dataset_reader=faiss_dataset_reader,
    faiss_dataset_wrapper=faiss_snli_wrapper,
    params_to_freeze=["_text_field_embedder.token_embedder_tokens.transformer_model"],
    # recur_depth=10,
)
