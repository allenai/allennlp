import glob
import itertools
from collections import defaultdict
from os import PathLike
from typing import Dict, List, Union, Optional, MutableMapping, NamedTuple, Tuple, Iterable
import json
import os
import re

from overrides import overrides
import torch
from torch import Tensor
from tqdm import tqdm
import torch.distributed as dist

from allennlp.common import util
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import int_to_device, lazy_groups_of
from allennlp.common.file_utils import cached_path, TensorCache
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector


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


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (re.search(comma_strip, inText) is not None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(",", "")
    return answer


def get_score(count: int) -> float:
    if count == 0:
        return 0.0
    elif count == 1:
        return 0.3
    elif count == 2:
        return 0.6
    elif count == 3:
        return 0.9
    elif count > 3:
        return 1.0
    else:
        raise ValueError()


@DatasetReader.register("vqav2")
class VQAv2Reader(DatasetReader):
    """
    Parameters
    ----------
    image_dir: `str`
        Path to directory containing `png` image files.
    image_featurizer: `GridEmbedder`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `RegionDetector`
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
        image_featurizer: GridEmbedder,
        region_detector: RegionDetector,
        *,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8
    ) -> None:
        super().__init__(
            max_instances=max_instances,
            manual_distributed_sharding=True,
            manual_multi_process_sharding=True,
        )

        if cuda_device is None:
            if torch.cuda.device_count() > 0:
                if util.is_distributed():
                    cuda_device = dist.get_rank() % torch.cuda.device_count()
                else:
                    cuda_device = 0
            else:
                cuda_device = -1
        check_for_gpu(cuda_device)
        self.cuda_device = int_to_device(cuda_device)

        self.images = {
            os.path.basename(filename): filename
            for filename in tqdm(
                glob.iglob(os.path.join(image_dir, "**", "*.jpg"), recursive=True),
                desc="Discovering images",
            )
        }

        # tokenizers and indexers
        if not tokenizer:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}
        self._token_indexers = token_indexers

        # image loading
        self.image_loader = image_loader
        self.image_featurizer = image_featurizer.to(self.cuda_device)
        self.region_detector = region_detector.to(self.cuda_device)

        # feature cache
        if feature_cache_dir is None:
            self._features_cache: MutableMapping[str, Tensor] = {}
            self._coordinates_cache: MutableMapping[str, Tensor] = {}
        else:
            os.makedirs(feature_cache_dir, exist_ok=True)
            self._features_cache = TensorCache(os.path.join(feature_cache_dir, "features"))
            self._coordinates_cache = TensorCache(os.path.join(feature_cache_dir, "coordinates"))

        self.image_processing_batch_size = image_processing_batch_size

    @overrides
    def _read(self, split: str):
        class Split(NamedTuple):
            annotations: Optional[str]
            questions: str

        splits = {
            "balanced_real_train": Split(
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip!v2_mscoco_train2014_annotations.json",
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip!v2_OpenEnded_mscoco_train2014_questions.json"
            ),
            "balanced_real_val": Split(
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip!v2_mscoco_val2014_annotations.json",
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip!v2_OpenEnded_mscoco_val2014_questions.json"
            ),
            "balanced_real_test": Split(
                None,
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip!v2_OpenEnded_mscoco_test2015_questions.json"
            ),
            "balanced_bas_train": Split(        # "bas" is Binary Abstract Scenes
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Binary_Train2017_abstract_v002.zip!abstract_v002_train2017_annotations.json",
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Binary_Train2017_abstract_v002.zip!OpenEnded_abstract_v002_train2017_questions.json"
            ),
            "balanced_bas_val": Split(
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Binary_Val2017_abstract_v002.zip!abstract_v002_val2017_annotations.json",
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Binary_Val2017_abstract_v002.zip!OpenEnded_abstract_v002_val2017_questions.json"
            ),
            "abstract_scenes_train": Split(
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Train_abstract_v002.zip!abstract_v002_train2015_annotations.json",
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Train_abstract_v002.zip!OpenEnded_abstract_v002_train2015_questions.json"
            ),
            "abstract_scenes_val": Split(
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Val_abstract_v002.zip!abstract_v002_val2015_annotations.json",
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Val_abstract_v002.zip!OpenEnded_abstract_v002_val2015_questions.json"
            ),
            "abstract_scenes_test": Split(
                None,
                "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Test_abstract_v002.zip!OpenEnded_abstract_v002_test2015_questions.json"
            )
        }

        try:
            split = splits[split]
        except KeyError:
            raise ValueError(
                f"Unrecognized split: {split}. We require a split, not a filename, for VQA "
                "because the image filenames require using the split."
            )

        if split.annotations is None:
            annotations_by_question_id = {}
        else:
            with open(cached_path(split.annotations, extract_archive=True)) as f:
                annotations = json.load(f)
                annotations_by_question_id = {
                    a["question_id"]: a
                    for a in annotations["annotations"]
                }
        with open(cached_path(split.questions, extract_archive=True)) as f:
            questions = json.load(f)

        # It would be much easier to just process one image at a time, but it's faster to process
        # them in batches. So this code gathers up instances until it has enough to fill up a batch
        # that needs processing, and then processes them all.
        question_dicts = self.shard_iterable(questions["questions"])
        question_dicts, processed_images = itertools.tee(question_dicts)
        processed_images = self._process_image_paths(
            self.images[f"{question_dict['image_id']:012d}.jpg"]
            for question_dict in question_dicts
        )

        for question_dict, processed_image in zip(question_dicts, processed_images):
            answers = annotations_by_question_id.get(question_dict["question_id"])
            if answers is not None:
                answers = answers["answers"]
            yield self.text_to_instance(question_dict["question"], processed_image, answers)

    def _process_image_paths(self, image_paths: Iterable[str]) -> Iterable[Tuple[Tensor, Tensor]]:
        batch = []
        unprocessed_paths = set()

        def yield_batch():
            # process the images
            paths = list(unprocessed_paths)
            images, sizes = self.image_loader(paths)
            with torch.no_grad():
                images = images.to(self.cuda_device)
                sizes = sizes.to(self.cuda_device)
                featurized_images = self.image_featurizer(images, sizes)
                detector_results = self.region_detector(images, sizes, featurized_images)
                features = detector_results["features"]
                coordinates = detector_results["coordinates"]

            # store the processed results in memory, so we can complete the batch
            paths_to_tensors = {
                path: (features[i], coordinates[i])
                for i, path in enumerate(paths)
            }

            # store the processed results in the cache
            for path, (features, coordinates) in paths_to_tensors.items():
                basename = os.path.basename(path)
                self._features_cache[basename] = features
                self._coordinates_cache[basename] = coordinates

            # yield the batch
            for b in batch:
                if isinstance(b, str):
                    yield paths_to_tensors[b]
                else:
                    yield b

        for image_path in image_paths:
            basename = os.path.basename(image_path)
            try:
                features = self._features_cache[basename]
                coordinates = self._coordinates_cache[basename]
                if len(batch) <= 0:
                    yield features, coordinates
                else:
                    batch.append((features, coordinates))
            except KeyError:
                batch.append(image_path)
                unprocessed_paths.add(image_path)
                if len(unprocessed_paths) >= self.image_processing_batch_size:
                    yield from yield_batch()
                    batch = []
                    unprocessed_paths = set()

        if len(batch) > 0:
            yield from yield_batch()


    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        image: Union[str, Tuple[Tensor, Tensor]],
        answers: List[Dict[str, str]] = None,
    ) -> Instance:
        tokenized_question = self._tokenizer.tokenize(question)
        question_field = TextField(tokenized_question, self._token_indexers)
        if isinstance(image, str):
            features, coords = next(self._process_image_paths([image]))
        else:
            features, coords = image

        fields = {
            "box_features": ArrayField(features),
            "box_coordinates": ArrayField(coords),
            "question": question_field,
        }

        if answers:
            answer_fields = []
            weights = []
            answer_counts: MutableMapping[str, int] = defaultdict(int)
            for answer_dict in answers:
                answer = preprocess_answer(answer_dict["answer"])
                answer_counts[answer] += 1

            for answer, count in answer_counts.items():
                # Using a namespace other than "labels" so that OOV answers don't crash.  We'll have
                # to mask OOV labels in the loss.  This is not ideal; it'd be better to remove OOV
                # answers from the training data entirely, but we can't do that in our current
                # pipeline without providing preprocessed input to the dataset reader.
                answer_fields.append(LabelField(answer, label_namespace="answers"))
                weights.append(get_score(count))

            fields["labels"] = ListField(answer_fields)
            fields["label_weights"] = ArrayField(torch.tensor(weights))

        return Instance(fields)
