import os
from os import PathLike
from typing import Dict, Union

from overrides import overrides

from allennlp.common.detectron import DetectronConfig, get_detectron_cfg
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance


@DatasetReader.register("nlvr2")
class Nlvr2Reader(DatasetReader):
    """
    Parameters
    ----------
    mask_prepositions_verbs: `bool`, optional (default=False)
        Whether to mask prepositions and verbs in each sentence
    drop_prepositions_verbs: `bool, optional (default=False)
        Whether to drop (remove without replacement) prepositions and verbs in each sentence
    lazy : `bool`, optional
        Whether to load data lazily. Passed to super class.
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        mask_prepositions_verbs: bool = False,
        drop_prepositions_verbs: bool = False,
        transformer_model: str = "bert-base-uncased",
        detectron_config: DetectronConfig = DetectronConfig(),
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)

        # find images
        import glob
        self.images = {
            os.path.basename(filename): filename
            for filename in glob.iglob(os.path.join(image_dir, "**", "*.png"), recursive=True)
        }

        # tokenizers and indexers
        from allennlp.data.tokenizers import PretrainedTransformerTokenizer
        self._tokenizer = PretrainedTransformerTokenizer(transformer_model)
        from allennlp.data.token_indexers import PretrainedTransformerIndexer
        from allennlp.data import TokenIndexer
        self._token_indexers: Dict[str, TokenIndexer] = {
            "tokens": PretrainedTransformerIndexer(transformer_model)
        }

        # old-school linguistics
        self.mask_prepositions_verbs = mask_prepositions_verbs
        self.drop_prepositions_verbs = drop_prepositions_verbs
        if mask_prepositions_verbs or drop_prepositions_verbs:
            # Loading Spacy to find prepositions and verbs
            import spacy
            self.spacy = spacy.load("en_core_web_sm")
        else:
            self.spacy = None

        # detectron
        if isinstance(detectron_config, DetectronConfig):
            detectron_config = detectron_config._asdict()
        cfg = get_detectron_cfg(**detectron_config)
        from allennlp.data.dataset_readers.dataset_utils.detectron_utils import DetectronProcessor
        self.detectron_processor = DetectronProcessor(cfg)

    @overrides
    def _read(self, split: str):
        nlvr_commit = "68a11a766624a5b665ec7594982b8ecbedc728c7"
        splits = {
            "dev": f"https://raw.githubusercontent.com/lil-lab/nlvr/{nlvr_commit}/nlvr2/data/dev.json",
            "test": f"https://raw.githubusercontent.com/lil-lab/nlvr/{nlvr_commit}/nlvr2/data/test1.json",
            "train": f"https://raw.githubusercontent.com/lil-lab/nlvr/{nlvr_commit}/nlvr2/data/train.json",
            "balanced_dev": f"https://raw.githubusercontent.com/lil-lab/nlvr/{nlvr_commit}/nlvr2/data/blanced/balanced_dev.json",
            "balanced_test": f"https://raw.githubusercontent.com/lil-lab/nlvr/{nlvr_commit}/nlvr2/data/blanced/balanced_test1.json",
            "unbalanced_dev": f"https://raw.githubusercontent.com/lil-lab/nlvr/{nlvr_commit}/nlvr2/data/blanced/unbalanced_dev.json",
            "unbalanced_test": f"https://raw.githubusercontent.com/lil-lab/nlvr/{nlvr_commit}/nlvr2/data/blanced/unbalanced_test1.json",
        }

        from allennlp.common.file_utils import cached_path
        json_file_path = cached_path(splits[split])

        from allennlp.common.file_utils import json_lines_from_file
        for json in json_lines_from_file(json_file_path):
            identifier = json["identifier"]
            sentence = json["sentence"]
            label = bool(json["label"])
            instance = self.text_to_instance(identifier, sentence, label)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        identifier: str,
        sentence: str,
        label: bool,
    ) -> Instance:
        if self.mask_prepositions_verbs:
            doc = self.spacy(sentence)
            prep_verb_starts = [
                (token.idx, len(token))
                for token in doc
                if token.dep_ == "prep" or token.pos_ == "VERB"
            ]
            new_sentence = ""
            prev_end = 0
            for (idx, length) in prep_verb_starts:
                new_sentence += sentence[prev_end:idx] + self._tokenizer.tokenizer.mask_token
                prev_end = idx + length
            new_sentence += sentence[prev_end:]
            sentence = new_sentence
        elif self.drop_prepositions_verbs:
            doc = self.spacy(sentence)
            prep_verb_starts = [
                (token.idx, len(token))
                for token in doc
                if token.dep_ == "prep" or token.pos_ == "VERB"
            ]
            new_sentence = ""
            prev_end = 0
            for (idx, length) in prep_verb_starts:
                new_sentence += sentence[prev_end:idx]
                prev_end = idx + length
            new_sentence += sentence[prev_end:]
            sentence = new_sentence
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        from allennlp.data.fields import TextField
        sentence_field = TextField(tokenized_sentence, self._token_indexers)

        # Load images
        image_name_base = identifier[:identifier.rindex("-")]
        images = [
            self.images[f"{image_name_base}-img{image_id}.png"]
            for image_id in [0, 1]
        ]
        visual_features = self.detectron_processor(images)
        from allennlp.data.fields import MetadataField

        fields = {
            "visual_features": ArrayField(visual_features),
            # "box_coordinates": images["instances/pred_boxes"],
            "sentence": MetadataField(sentence),
            "identifier": MetadataField(identifier),
            "sentence_field": sentence_field,
        }
        return Instance(fields)
