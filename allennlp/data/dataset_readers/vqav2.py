import logging
from collections import Counter
from os import PathLike
from typing import (
    Dict,
    List,
    Union,
    Optional,
    MutableMapping,
    NamedTuple,
    Tuple,
    Iterable,
)
import json
import re

from overrides import overrides
import torch
from torch import Tensor

from allennlp.common.lazy import Lazy
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ArrayField, LabelField, ListField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector
from allennlp.data.dataset_readers.vision_reader import VisionReader

logger = logging.getLogger(__name__)

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile(r"(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def process_punctuation(inText: str) -> str:
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (re.search(comma_strip, inText) is not None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(input: str) -> str:
    output = []
    for word in input.lower().split():
        word = manual_map.get(word, word)
        if word not in articles:
            output.append(word)
        else:
            pass
    for index, word in enumerate(output):
        if word in contractions:
            output[index] = contractions[word]
    return " ".join(output)


def preprocess_answer(answer: str) -> str:
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(",", "")
    return answer


def get_score(count: int) -> float:
    return min(1.0, count / 3)


@DatasetReader.register("vqav2")
class VQAv2Reader(VisionReader):
    """
    Parameters
    ----------
    image_dir: `str`
        Path to directory containing `png` image files.
    image_loader : `ImageLoader`
    image_featurizer: `Lazy[GridEmbedder]`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `Lazy[RegionDetector]`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    data_dir: `str`
        Path to directory containing text files for each dataset split. These files contain
        the sentences and metadata for each task instance.
    tokenizer: `Tokenizer`, optional
    token_indexers: `Dict[str, TokenIndexer]`
    lazy : `bool`, optional
        Whether to load data lazily. Passed to super class.
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        image_loader: ImageLoader,
        image_featurizer: Lazy[GridEmbedder],
        region_detector: Lazy[RegionDetector],
        *,
        answer_vocab: Union[
            Vocabulary, str
        ] = "https://storage.googleapis.com/allennlp-public-data/vqav2/vqav2_vocab.tar.gz",
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        skip_image_feature_extraction: bool = False,
        keep_unanswerable_questions: bool = True,
    ) -> None:
        super().__init__(
            image_dir,
            image_loader,
            image_featurizer,
            region_detector,
            feature_cache_dir=feature_cache_dir,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            cuda_device=cuda_device,
            max_instances=max_instances,
            image_processing_batch_size=image_processing_batch_size,
            skip_image_feature_extraction=skip_image_feature_extraction,
        )

        # read answer vocab
        if keep_unanswerable_questions:
            self.answer_vocab = None
        else:
            if isinstance(answer_vocab, str):
                answer_vocab = cached_path(answer_vocab, extract_archive=True)
                answer_vocab = Vocabulary.from_files(answer_vocab)
            self.answer_vocab = frozenset(
                preprocess_answer(a)
                for a in answer_vocab.get_token_to_index_vocabulary("answers").keys()
            )

    @overrides
    def _read(self, splits_or_list_of_splits: Union[str, List[str]]):
        # if we are given a list of splits, concatenate them
        if isinstance(splits_or_list_of_splits, str):
            split_name = splits_or_list_of_splits
        else:
            for split_name in splits_or_list_of_splits:
                yield from self._read(split_name)
            return

        # if the splits are using slicing syntax, honor it
        slice_match = re.match(r"(.*)\[([0123456789:]*)]", split_name)
        if slice_match is None:
            question_slice = slice(None, None, None)
        else:
            split_name = slice_match[1]
            slice_args = [int(a) if len(a) > 0 else None for a in slice_match[2].split(":")]
            question_slice = slice(*slice_args)

        class Split(NamedTuple):
            annotations: Optional[str]
            questions: str

        aws_base = "https://s3.amazonaws.com/cvmlp/vqa/"
        mscoco_base = aws_base + "mscoco/vqa/"
        scene_base = aws_base + "abstract_v002/vqa/"

        # fmt: off
        splits = {
            "balanced_real_train": Split(
                mscoco_base + "v2_Annotations_Train_mscoco.zip!v2_mscoco_train2014_annotations.json",  # noqa: E501
                mscoco_base + "v2_Questions_Train_mscoco.zip!v2_OpenEnded_mscoco_train2014_questions.json",  # noqa: E501
            ),
            "balanced_real_val": Split(
                mscoco_base + "v2_Annotations_Val_mscoco.zip!v2_mscoco_val2014_annotations.json",  # noqa: E501
                mscoco_base + "v2_Questions_Val_mscoco.zip!v2_OpenEnded_mscoco_val2014_questions.json",  # noqa: E501
            ),
            "balanced_real_test": Split(
                None,
                mscoco_base + "v2_Questions_Test_mscoco.zip!v2_OpenEnded_mscoco_test2015_questions.json",  # noqa: E501
            ),
            "balanced_bas_train": Split(  # "bas" is Binary Abstract Scenes
                scene_base + "Annotations_Binary_Train2017_abstract_v002.zip!abstract_v002_train2017_annotations.json",  # noqa: E501
                scene_base + "Questions_Binary_Train2017_abstract_v002.zip!OpenEnded_abstract_v002_train2017_questions.json",  # noqa: E501
            ),
            "balanced_bas_val": Split(
                scene_base + "Annotations_Binary_Val2017_abstract_v002.zip!abstract_v002_val2017_annotations.json",  # noqa: E501
                scene_base + "Questions_Binary_Val2017_abstract_v002.zip!OpenEnded_abstract_v002_val2017_questions.json",  # noqa: E501
            ),
            "abstract_scenes_train": Split(
                scene_base + "Annotations_Train_abstract_v002.zip!abstract_v002_train2015_annotations.json",  # noqa: E501
                scene_base + "Questions_Train_abstract_v002.zip!OpenEnded_abstract_v002_train2015_questions.json",  # noqa: E501
            ),
            "abstract_scenes_val": Split(
                scene_base + "Annotations_Val_abstract_v002.zip!abstract_v002_val2015_annotations.json",  # noqa: E501
                scene_base + "Questions_Val_abstract_v002.zip!OpenEnded_abstract_v002_val2015_questions.json",  # noqa: E501
            ),
            "abstract_scenes_test": Split(
                None,
                scene_base + "Questions_Test_abstract_v002.zip!OpenEnded_abstract_v002_test2015_questions.json",  # noqa: E501
            ),
            "unittest": Split(
                "test_fixtures/data/vqav2/annotations.json",
                "test_fixtures/data/vqav2/questions.json"
            )
        }
        # fmt: on

        try:
            split = splits[split_name]
        except KeyError:
            raise ValueError(
                f"Unrecognized split: {split_name}. We require a split, not a filename, for "
                "VQA because the image filenames require using the split."
            )

        annotations_by_question_id = {}
        if split.annotations is not None:
            with open(cached_path(split.annotations, extract_archive=True)) as f:
                annotations = json.load(f)
            for a in annotations["annotations"]:
                annotations_by_question_id[a["question_id"]] = a

        questions = []
        with open(cached_path(split.questions, extract_archive=True)) as f:
            questions_file = json.load(f)
        image_subtype = questions_file["data_subtype"]
        for ques in questions_file["questions"]:
            ques["image_subtype"] = image_subtype
            questions.append(ques)
        questions = questions[question_slice]

        question_dicts = list(self.shard_iterable(questions))
        processed_images: Iterable[Optional[Tuple[Tensor, Tensor]]]
        if not self.skip_image_feature_extraction:
            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.
            processed_images = self._process_image_paths(
                self.images[
                    f"COCO_{question_dict['image_subtype']}_{question_dict['image_id']:012d}.jpg"
                ]
                for question_dict in question_dicts
            )
        else:
            processed_images = [None for i in range(len(question_dicts))]

        attempted_instances_count = 0
        failed_instances_count = 0
        for question_dict, processed_image in zip(question_dicts, processed_images):
            answers = annotations_by_question_id.get(question_dict["question_id"])
            if answers is not None:
                answers = answers["answers"]

            instance = self.text_to_instance(question_dict["question"], processed_image, answers)
            attempted_instances_count += 1
            if instance is None:
                failed_instances_count += 1
            else:
                yield instance

            if attempted_instances_count % 2000 == 0:
                failed_instances_fraction = failed_instances_count / attempted_instances_count
                if failed_instances_fraction > 0.1:
                    logger.warning(
                        f"{failed_instances_fraction*100:.0f}% of instances have no answers."
                    )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        image: Union[str, Tuple[Tensor, Tensor]],
        answers: Optional[List[Dict[str, str]]] = None,
        *,
        use_cache: bool = True,
    ) -> Optional[Instance]:
        tokenized_question = self._tokenizer.tokenize(question)
        question_field = TextField(tokenized_question, None)

        fields: Dict[str, Field] = {
            "question": question_field,
        }

        if image is not None:
            if isinstance(image, str):
                features, coords = next(self._process_image_paths([image], use_cache=use_cache))
            else:
                features, coords = image

            fields["box_features"] = ArrayField(features)
            fields["box_coordinates"] = ArrayField(coords)
            fields["box_mask"] = ArrayField(
                features.new_full((features.shape[0],), True, dtype=torch.bool),
                padding_value=False,
                dtype=torch.bool,
            )

        if answers:
            answer_fields = []
            weights = []
            answer_counts: MutableMapping[str, int] = Counter()

            for answer in (a["answer"] for a in answers):
                answer_counts[preprocess_answer(answer)] += 1

            for answer, count in answer_counts.items():
                if self.answer_vocab is None or answer in self.answer_vocab:
                    answer_fields.append(LabelField(answer, label_namespace="answers"))
                    weights.append(get_score(count))

            if len(answer_fields) <= 0:
                return None

            fields["labels"] = ListField(answer_fields)
            fields["label_weights"] = ArrayField(torch.tensor(weights))

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["question"].token_indexers = self._token_indexers  # type: ignore
