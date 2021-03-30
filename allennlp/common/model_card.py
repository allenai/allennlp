"""
A specification for defining model cards as described in
[Model Cards for Model Reporting (Mitchell et al, 2019)]
(https://api.semanticscholar.org/CorpusID:52946140)

The descriptions of the fields and some examples
are taken from the paper.

The specification is provided to prompt model developers
to think about the various aspects that should ideally
be reported. The information filled should adhere to
the spirit of transparency rather than the letter; i.e.,
it should not be filled for the sake of being filled. If
the information cannot be inferred, it should be left empty.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Callable
from allennlp.common.from_params import FromParams

from allennlp.models import Model
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


def get_description(model_class):
    """
    Returns the model's description from the docstring.
    """
    return model_class.__doc__.split("# Parameters")[0].strip()


class ModelCardInfo(FromParams):
    def to_dict(self):
        """
        Only the non-empty attributes are returned, to minimize empty values.
        """
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info[key] = val
        return info

    def __str__(self):
        display = ""
        for key, val in self.to_dict().items():
            display += "\n" + key.replace("_", " ").capitalize() + ": "
            display += "\n\t" + val.replace("\n", "\n\t") + "\n"
        if not display:
            display = super(ModelCardInfo, self).__str__()
        return display.strip()


@dataclass(frozen=True)
class Paper(ModelCardInfo):
    """
    This provides information about the paper.

    # Parameters

    title : `str`
        The name of the paper.

    url : `str`
        A web link to the paper.

    citation : `str`
        The BibTex for the paper.

    """

    title: Optional[str] = None
    url: Optional[str] = None
    citation: Optional[str] = None


class ModelDetails(ModelCardInfo):
    """
    This provides the basic information about the model.

    # Parameters

    description : `str`
        A high-level overview of the model.
        Eg. The model implements a reading comprehension model patterned
            after the proposed model in [Devlin et al, 2018]
            (https://api.semanticscholar.org/CorpusID:52967399), with improvements
            borrowed from the SQuAD model in the transformers project.
            It predicts start tokens and end tokens with a linear layer on top of
            word piece embeddings.

    short_description : `str`
        A one-line description of the model.
        Eg. A reading comprehension model patterned after RoBERTa,
            with improvements borrowed from the SQuAD model in the transformers project.

    developed_by : `str`
        Person/organization that developed the model. This can be used by all
        stakeholders to infer details pertaining to model development and
        potential conflicts of interest.

    contributed_by : `str`
        Person that contributed the model to the repository.

    date : `str`
        The date on which the model was contributed. This is useful for all
        stakeholders to become further informed on what techniques and
        data sources were likely to be available during model development.
        Format example: 2020-09-23

    version : `str`
        The version of the model, and how it differs from previous versions.
        This is useful for all stakeholders to track whether the model is the
        latest version, associate known bugs to the correct model versions,
        and aid in model comparisons.

    model_type : `str`
        The type of the model; the basic architecture. This is likely to be
        particularly relevant for software and model developers, as well as
        individuals knowledgeable about machine learning, to highlight what
        kinds of assumptions are encoded in the system.
        Eg. Naive Bayes Classifier.

    paper : `Union[str, Dict, Paper]`
        The paper on which the model is based.
        Format example:
        {
            "title": "Model Cards for Model Reporting (Mitchell et al, 2019)",
            "url": "https://api.semanticscholar.org/CorpusID:52946140",
            "citation": "<BibTex>",
        }

    license : `str`
        License information for the model.

    contact : `str`
        The email address to reach out to the relevant developers/contributors
        for questions/feedback about the model.

    """

    def __init__(
        self,
        description: Optional[str] = None,
        short_description: Optional[str] = None,
        developed_by: Optional[str] = None,
        contributed_by: Optional[str] = None,
        date: Optional[str] = None,
        version: Optional[str] = None,
        model_type: Optional[str] = None,
        paper: Optional[Union[str, Dict, Paper]] = None,
        license: Optional[str] = None,
        contact: Optional[str] = None,
    ):
        self.description = description
        self.short_description = short_description
        self.developed_by = developed_by
        self.contributed_by = contributed_by
        self.date = date
        self.version = version
        self.model_type = model_type
        if isinstance(paper, Paper):
            self.paper = paper
        elif isinstance(paper, Dict):
            self.paper = Paper(**paper)
        else:
            self.paper = Paper(title=paper)
        self.license = license
        self.contact = contact


@dataclass(frozen=True)
class IntendedUse(ModelCardInfo):
    """
    This determines what the model should and should not be used for.

    # Parameters

    primary_uses : `str`
        Details the primary intended uses of the model; whether it was developed
        for general or specific tasks.
        Eg. The toxic text identifier model was developed to identify
            toxic comments on online platforms. An example use case is
            to provide feedback to comment authors.

    primary_users : `str`
        The primary intended users. For example, was the model developed
        for entertainment purposes, for hobbyists, or enterprise solutions?
        This helps users gain insight into how robust the model may be to
        different kinds of inputs.

    out_of_scope_use_cases : `str`
        Highlights the technology that the model might easily be confused with,
        or related contexts that users could try to apply the model to.
        Eg. the toxic text identifier model is not intended for fully automated
            moderation, or to make judgements about specific individuals.

        Also recommends a related or similar model that was designed to better
        meet a particular need, where possible.
        Eg. not for use on text examples longer than 100 tokens; please use
        the bigger-toxic-text-identifier instead.
    """

    primary_uses: Optional[str] = None
    primary_users: Optional[str] = None
    out_of_scope_use_cases: Optional[str] = None


@dataclass(frozen=True)
class Factors(ModelCardInfo):
    """
    This provides a summary of relevant factors such as
    demographics, instrumentation used, etc. for which the
    model performance may vary.

    # Parameters

    relevant_factors : `str`
         The foreseeable salient factors for which model performance may vary,
         and how these were determined.
         Eg. the model performance may vary for variations in dialects of English.

    evaluation_factors : `str`
        Mentions the factors that are being reported, and the reasons for why
        they were chosen. Also includes the reasons for choosing different
        evaluation factors than relevant factors.

        Eg. While dialect variation is a relevant factor,
        dialect-specific annotations were not available, and hence, the
        performance was not evaluated on different dialects.
    """

    relevant_factors: Optional[str] = None
    evaluation_factors: Optional[str] = None


@dataclass(frozen=True)
class Metrics(ModelCardInfo):
    """
    This lists the reported metrics and the reasons
    for choosing them.

    # Parameters

    model_performance_measures : `str`
        Which model performance measures were selected and the reasons for
        selecting them.
    decision_thresholds : `str`
        If decision thresholds are used, what are they, and the reasons for
        choosing them.
    variation_approaches : `str`
        How are the measurements and estimations of these metrics calculated?
        Eg. standard deviation, variance, confidence intervals, KL divergence.
        Details of how these values are approximated should also be included.
        Eg. average of 5 runs, 10-fold cross-validation, etc.
    """

    model_performance_measures: Optional[str] = None
    decision_thresholds: Optional[str] = None
    variation_approaches: Optional[str] = None


@dataclass(frozen=True)
class Dataset(ModelCardInfo):
    """
    This provides basic information about the dataset.

    # Parameters

    name : `str`
        The name of the dataset.

    url : `str`
        A web link to the dataset information/datasheet.

    processed_url : `str`
        A web link to a downloadable/directly usable version
        of the dataset, if available.

    notes: `str`
        Any other notes on downloading/processing the data.
    """

    name: Optional[str] = None
    url: Optional[str] = None
    processed_url: Optional[str] = None
    notes: Optional[str] = None


class EvaluationData(ModelCardInfo):
    """
    This provides information about the evaluation data.

    # Parameters

    dataset : `Union[str, Dict, Dataset]`
        The name(s) (and link(s), if available) of the dataset(s) used to evaluate
        the model. Optionally, provide a link to the relevant datasheet(s) as well.
    motivation : `str`
        The reasons for selecting the dataset(s).
        Eg. For the BERT model, document-level corpora were used rather than a
            shuffled sentence-level corpus in order to extract long contiguous sequences.
    preprocessing : `str`
        How was the data preprocessed for evaluation?
        Eg. tokenization of sentences, filtering of paragraphs by length, etc.
    """

    def __init__(
        self,
        dataset: Optional[Union[str, Dict, Dataset]] = None,
        motivation: Optional[str] = None,
        preprocessing: Optional[str] = None,
    ):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, Dict):
            self.dataset = Dataset(**dataset)
        else:
            self.dataset = Dataset(name=dataset)
        self.motivation = motivation
        self.preprocessing = preprocessing

    def to_dict(self):
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info["evaluation_" + key] = val
        return info


class TrainingData(ModelCardInfo):
    """
    This provides information about the training data. If the model was initialized
    from pretrained weights, a link to the pretrained model's model card/training
    data can additionally be provided, if available. Any relevant definitions should
    also be included.

    # Parameters

    dataset : `Union[str, Dict, Dataset]`
        The name(s) (and link(s), if available) of the dataset(s) used to train
        the model. Optionally, provide a link to the relevant datasheet(s) as well.
        Eg. * Proprietary data from Perspective API; includes comments from online
              forums such as Wikipedia and New York Times, with crowdsourced labels of
              whether the comment is "toxic".
            * "Toxic" is defined as "a rude, disrespectful, or unreasonable comment
              that is likely to make you leave a discussion."
    motivation : `str`
        The reasons for selecting the dataset(s).
        Eg. For the BERT model, document-level corpora were used rather than a
            shuffled sentence-level corpus in order to extract long contiguous sequences.
    preprocessing : `str`
        Eg. Only the text passages were extracted from English Wikipedia;  lists, tables,
            and headers were ignored.
    """

    def __init__(
        self,
        dataset: Optional[Union[str, Dict, Dataset]] = None,
        motivation: Optional[str] = None,
        preprocessing: Optional[str] = None,
    ):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, Dict):
            self.dataset = Dataset(**dataset)
        else:
            self.dataset = Dataset(name=dataset)
        self.motivation = motivation
        self.preprocessing = preprocessing

    def to_dict(self):
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info["training_" + key] = val
        return info


@dataclass(frozen=True)
class QuantitativeAnalyses(ModelCardInfo):
    """
    This provides disaggregated evaluation of how the
    model performed based on chosen metrics, with confidence
    intervals, if possible. Links to plots/figures showing
    the metrics can also be provided.

    # Parameters

    unitary_results : `str`
        The performance of the model with respect to each chosen
        factor.
    intersectional_results : `str`
        The performance of the model with respect to the intersection
        of the evaluated factors.
    """

    unitary_results: Optional[str] = None
    intersectional_results: Optional[str] = None


@dataclass(frozen=True)
class ModelEthicalConsiderations(ModelCardInfo):
    """
    This highlights any ethical considerations to keep
    in mind when using the model.
    Eg. Is the model intended to be used for informing
    decisions on human life? Does it use sensitive data?
    What kind of risks are possible, and what mitigation
    strategies were used to address them?
    Eg. The model does not take into account user history
        when making judgments about toxicity, due to privacy
        concerns.
    """

    ethical_considerations: Optional[str] = None


@dataclass(frozen=True)
class ModelCaveatsAndRecommendations(ModelCardInfo):
    """
    This lists any additional concerns. For instance, were any
    relevant groups not present in the evaluation data?
    Eg. The evaluation data is synthetically designed to be
        representative of common use cases and concerns, but
        may not be comprehensive.
    """

    caveats_and_recommendations: Optional[str] = None


class ModelUsage(ModelCardInfo):
    """
    archive_file : `str`, optional
        The location of model's pretrained weights.
    training_config : `str`, optional
        A url to the training config.
    install_instructions : `str`, optional
        Any additional instructions for installations.
    overrides : `Dict`, optional
        Optional overrides for the model's architecture.
    """

    _storage_location = "https://storage.googleapis.com/allennlp-public-models/"
    _config_location = (
        "https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config"
    )

    def __init__(
        self,
        archive_file: Optional[str] = None,
        training_config: Optional[str] = None,
        install_instructions: Optional[str] = None,
        overrides: Optional[Dict] = None,
    ):

        if archive_file and not archive_file.startswith("https:"):
            archive_file = os.path.join(self._storage_location, archive_file)

        if training_config and not training_config.startswith("https:"):
            training_config = os.path.join(self._config_location, training_config)

        self.archive_file = archive_file
        self.training_config = training_config
        self.install_instructions = install_instructions
        self.overrides = overrides


class ModelCard(ModelCardInfo):
    """
    The model card stores the recommended attributes for model reporting.

    # Parameters

    id : `str`
        Model's id, following the convention of task-model-relevant-details.
        Example: rc-bidaf-elmo for a reading comprehension BiDAF model using ELMo embeddings.
    registered_model_name : `str`, optional
        The model's registered name. If `model_class` is not given, this will be used
        to find any available `Model` registered with this name.
    model_class : `type`, optional
        If given, the `ModelCard` will pull some default information from the class.
    registered_predictor_name : `str`, optional
        The registered name of the corresponding predictor.
    display_name : `str`, optional
        The pretrained model's display name.
    task_id : `str`, optional
        The id of the task for which the model was built.
    model_usage: `Union[ModelUsage, str]`, optional
    model_details : `Union[ModelDetails, str]`, optional
    intended_use : `Union[IntendedUse, str]`, optional
    factors : `Union[Factors, str]`, optional
    metrics : `Union[Metrics, str]`, optional
    evaluation_data : `Union[EvaluationData, str]`, optional
    quantitative_analyses : `Union[QuantitativeAnalyses, str]`, optional
    ethical_considerations : `Union[ModelEthicalConsiderations, str]`, optional
    caveats_and_recommendations : `Union[ModelCaveatsAndRecommendations, str]`, optional

    !!! Note
        For all the fields that are `Union[ModelCardInfo, str]`, a `str` input will be
        treated as the first argument of the relevant constructor.

    """

    def __init__(
        self,
        id: str,
        registered_model_name: Optional[str] = None,
        model_class: Optional[Callable[..., Model]] = None,
        registered_predictor_name: Optional[str] = None,
        display_name: Optional[str] = None,
        task_id: Optional[str] = None,
        model_usage: Optional[Union[str, ModelUsage]] = None,
        model_details: Optional[Union[str, ModelDetails]] = None,
        intended_use: Optional[Union[str, IntendedUse]] = None,
        factors: Optional[Union[str, Factors]] = None,
        metrics: Optional[Union[str, Metrics]] = None,
        evaluation_data: Optional[Union[str, EvaluationData]] = None,
        training_data: Optional[Union[str, TrainingData]] = None,
        quantitative_analyses: Optional[Union[str, QuantitativeAnalyses]] = None,
        model_ethical_considerations: Optional[Union[str, ModelEthicalConsiderations]] = None,
        model_caveats_and_recommendations: Optional[
            Union[str, ModelCaveatsAndRecommendations]
        ] = None,
    ):

        assert id
        if not model_class and registered_model_name:
            try:
                model_class = Model.by_name(registered_model_name)
            except ConfigurationError:
                logger.warning("{} is not a registered model.".format(registered_model_name))

        if model_class:
            display_name = display_name or model_class.__name__
            model_details = model_details or get_description(model_class)
            if not registered_predictor_name:
                registered_predictor_name = model_class.default_predictor  # type: ignore

        if isinstance(model_usage, str):
            model_usage = ModelUsage(archive_file=model_usage)
        if isinstance(model_details, str):
            model_details = ModelDetails(description=model_details)
        if isinstance(intended_use, str):
            intended_use = IntendedUse(primary_uses=intended_use)
        if isinstance(factors, str):
            factors = Factors(relevant_factors=factors)
        if isinstance(metrics, str):
            metrics = Metrics(model_performance_measures=metrics)
        if isinstance(evaluation_data, str):
            evaluation_data = EvaluationData(dataset=evaluation_data)
        if isinstance(training_data, str):
            training_data = TrainingData(dataset=training_data)
        if isinstance(quantitative_analyses, str):
            quantitative_analyses = QuantitativeAnalyses(unitary_results=quantitative_analyses)
        if isinstance(model_ethical_considerations, str):
            model_ethical_considerations = ModelEthicalConsiderations(model_ethical_considerations)
        if isinstance(model_caveats_and_recommendations, str):
            model_caveats_and_recommendations = ModelCaveatsAndRecommendations(
                model_caveats_and_recommendations
            )

        self.id = id
        self.registered_model_name = registered_model_name
        self.registered_predictor_name = registered_predictor_name
        self.display_name = display_name
        self.task_id = task_id
        self.model_usage = model_usage
        self.model_details = model_details
        self.intended_use = intended_use
        self.factors = factors
        self.metrics = metrics
        self.evaluation_data = evaluation_data
        self.training_data = training_data
        self.quantitative_analyses = quantitative_analyses
        self.model_ethical_considerations = model_ethical_considerations
        self.model_caveats_and_recommendations = model_caveats_and_recommendations

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the `ModelCard` to a flat dictionary object. This can be converted to
        json and passed to any front-end.
        """
        info = {}
        for key, val in self.__dict__.items():
            if key != "id":
                if isinstance(val, ModelCardInfo):
                    info.update(val.to_dict())
                else:
                    if val is not None:
                        info[key] = val
        return info
