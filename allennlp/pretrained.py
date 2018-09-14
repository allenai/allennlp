# pylint: disable=line-too-long,invalid-name

from allennlp import predictors
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

class PretrainedModel:
    """
    A pretrained model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor.
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)

# TODO(Mark): Figure out a way to make PretrainedModel generic on Predictor, so we can remove these type ignores.

#### Models in the demo ####
def srl_with_elmo_luheng_2018() -> predictors.SemanticRoleLabelerPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
                            'semantic-role-labeling')
    return model.predictor() # type: ignore

def bidirectional_attention_flow_seo_2017() -> predictors.BidafPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz',
                            'machine-comprehension')
    return model.predictor() # type: ignore

def open_information_extraction_stanovsky_2018() -> predictors.OpenIePredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz',
                            'open-information-extraction')
    return model.predictor() # type: ignore

def decomposable_attention_with_elmo_parikh_2017() -> predictors.DecomposableAttentionPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz',
                            'textual-entailment')
    return model.predictor() # type: ignore

def neural_coreference_resolution_lee_2017() -> predictors.CorefPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',
                            'coreference-resolution')
    return model.predictor() # type: ignore

def named_entity_recognition_with_elmo_peters_2018() -> predictors.SentenceTaggerPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.30.tar.gz',
                            'sentence-tagger')
    return model.predictor() # type: ignore

def fine_grained_named_entity_recognition_with_elmo_peters_2018() -> predictors.SentenceTaggerPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.08.31.tar.gz',
                            'sentence-tagger')
    return model.predictor() # type: ignore

def span_based_constituency_parsing_with_elmo_joshi_2018() -> predictors.ConstituencyParserPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz',
                            'constituency-parser')
    return model.predictor() # type: ignore

def biaffine_parser_stanford_dependencies_todzat_2017() -> predictors.BiaffineDependencyParserPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz',
                            'biaffine-dependency-parser')
    return model.predictor() # type: ignore


#### Models not in the demo ####
def biaffine_parser_universal_dependencies_todzat_2017() -> predictors.BiaffineDependencyParserPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ud-2018.08.23.tar.gz',
                            'biaffine-dependency-parser')
    return model.predictor() # type: ignore

def esim_nli_with_elmo_chen_2017() -> predictors.DecomposableAttentionPredictor:
    model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/esim-elmo-2018.05.17.tar.gz',
                            'textual-entailment')
    return model.predictor() # type: ignore
