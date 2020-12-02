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

from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, TextField
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
    image_loader: `ImageLoader`
        The image loader component used to load the images.
    image_featurizer: `GridEmbedder`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `RegionDetector`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    answer_vocab: `Union[Vocabulary, str]`, optional
        The vocabulary to use for answers. The reader will look into the `"answers"` namespace
        in the vocabulary to find possible answers.
        If this is given, the reader only outputs instances with answers contained in this vocab.
        If this is not given, the reader outputs all instances with all answers.
        If this is a URL or filename, we will download a previously saved vocabulary from there.
    feature_cache_dir: `Union[str, PathLike]`, optional
        An optional directory to cache the featurized images in. Featurizing images takes a long
        time, and many images are duplicated, so we highly recommend to use this cache.
    tokenizer: `Tokenizer`, optional
        The `Tokenizer` to use to tokenize the text. By default, this uses the tokenizer for
        `"bert-base-uncased"`.
    token_indexers: `Dict[str, TokenIndexer]`, optional
        The `TokenIndexer` to use. By default, this uses the indexer for `"bert-base-uncased"`.
    cuda_device: `Union[int, torch.device]`, optional
        Either a torch device or a GPU number. This is the GPU we'll use to featurize the images.
    max_instances: `int`, optional
        For debugging, you can use this parameter to limit the number of instances the reader
        returns.
    image_processing_batch_size: `int`
        The number of images to process at one time while featurizing. Default is 8.
    run_image_feature_extraction: `bool`
        If this is set to `False`, we skip featurizing images completely. This can be useful
        for debugging or for generating the vocabulary ahead of time. Default is `True`.
    multiple_answers_per_question: `bool`
        VQA questions have multiple answers. By default, we use all of them, and give more
        points to the more common answer. But VQA also has a special answer, the so-called
        "multiple choice answer". If this is set to `False`, we only use that answer.
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        image_loader: ImageLoader,
        image_featurizer: GridEmbedder,
        region_detector: RegionDetector,
        *,
        answer_vocab: Optional[Union[Vocabulary, str]] = None,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        run_image_feature_extraction: bool = True,
        multiple_answers_per_question: bool = True
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
            run_image_feature_extraction=run_image_feature_extraction,
        )

        # read answer vocab
        if answer_vocab is None:
            self.answer_vocab = None
        else:
            if isinstance(answer_vocab, str):
                answer_vocab = cached_path(answer_vocab, extract_archive=True)
                answer_vocab = Vocabulary.from_files(answer_vocab)
            self.answer_vocab = frozenset(
                preprocess_answer(a)
                for a in answer_vocab.get_token_to_index_vocabulary("answers").keys()
            )

        if run_image_feature_extraction:
            # normalize self.images some more
            # At this point, self.images maps filenames to full paths, but we want to map image ids to full paths.
            filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

            def id_from_filename(filename: str) -> Optional[int]:
                match = filename_re.fullmatch(filename)
                if match is None:
                    return None
                return int(match.group(1))

            self.images = {id_from_filename(name): full_path for name, full_path in self.images.items()}
            if None in self.images:
                del self.images[None]

        self.multiple_answers_per_question = multiple_answers_per_question

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
            raise ValueError(f"Unrecognized split: {split_name}.")

        answers_by_question_id = {}
        if split.annotations is not None:
            with open(cached_path(split.annotations, extract_archive=True)) as f:
                annotations = json.load(f)
            for a in annotations["annotations"]:
                qid = a["question_id"]
                answer_counts: MutableMapping[str, int] = Counter()
                if self.multiple_answers_per_question:
                    for answer in (answer_dict["answer"] for answer_dict in a["answers"]):
                        answer_counts[preprocess_answer(answer)] += 1
                else:
                    answer_counts[preprocess_answer(a["multiple_choice_answer"])] = 1
                answers_by_question_id[qid] = answer_counts

        questions = []
        with open(cached_path(split.questions, extract_archive=True)) as f:
            questions_file = json.load(f)
        for ques in questions_file["questions"]:
            questions.append(ques)
        questions = questions[question_slice]

        question_dicts = list(self.shard_iterable(questions))
        processed_images: Iterable[Optional[Tuple[Tensor, Tensor]]]
        if self.run_image_feature_extraction:
            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.
            processed_images = self._process_image_paths(
                self.images[int(question_dict["image_id"])] for question_dict in question_dicts
            )
        else:
            processed_images = [None for i in range(len(question_dicts))]

        attempted_instances_count = 0
        failed_instances_count = 0
        for question_dict, processed_image in zip(question_dicts, processed_images):
            answers = answers_by_question_id.get(question_dict["question_id"])

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
        answer_counts: Optional[MutableMapping[str, int]] = None,
        *,
        use_cache: bool = True,
    ) -> Optional[Instance]:
        tokenized_question = self._tokenizer.tokenize(question)
        question_field = TextField(tokenized_question, None)
        from allennlp.data import Field

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

        if answer_counts is not None:
            answer_fields = []
            weights = []

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
