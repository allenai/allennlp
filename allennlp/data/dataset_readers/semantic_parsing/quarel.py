"""
Reader for QuaRel dataset
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import re

import numpy as np
from overrides import overrides

import tqdm

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import ArrayField, Field, TextField, KnowledgeGraphField, LabelField
from allennlp.data.fields import IndexField, ListField, MetadataField, ProductionRuleField
from allennlp.data.fields import SequenceLabelField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph
from allennlp.semparse.contexts.quarel_utils import WorldTaggerExtractor, words_from_entity_string
from allennlp.semparse.contexts.quarel_utils import LEXICAL_CUES, align_entities
from allennlp.semparse.worlds.quarel_world import QuarelWorld


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("quarel")
class QuarelDatasetReader(DatasetReader):
    """
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    replace_world_entities : ``bool`` (optional, default=False)
        Replace world entities (w/stemming) with "worldone" and "worldtwo" directly in the question
    world_extraction_model: ``str`` (optional, default=None)
        Reference (file or URL) to world tagger model used to extract worlds.
    align_world_extractions : ``bool`` (optional, default=False)
        Use alignment of extracted worlds with gold worlds, to pick the appropriate gold LF.
    gold_world_extractions : ``bool`` (optional, default=False)
        Use gold worlds rather than world extractor
    tagger_only : ``bool`` (optional default=False)
        Only output tagging information, in format for CRF tagger
    denotation_only: ``bool`` (optional, default=False)
        Only output information needed for a denotation-only model (no LF)
    entity_bits_mode : ``str`` (optional, default=None)
        If set, add a field for entity bits ("simple" = 1.0 value for world1 and world2,
        "simple_collapsed" = single 1.0 value for any world).
    entity_types : ``List[str]`` (optional, default=None)
        List of entity types used for tagger model
    world_extraction_model : ``str`` (optional, default=None)
        Reference to model file for world tagger model
    lexical_cues : ``List[str]`` (optional, default=None)
        List of lexical cue categories to include when using dynamic attributes
    skip_attributes_regex: ``str`` (optional, default=None)
        Regex string for which examples and attributes to skip if the LF matches
    lf_syntax: ``str``
        Which logical form formalism to use
    """
    def __init__(self,
                 lazy: bool = False,
                 sample: int = -1,
                 lf_syntax: str = None,
                 replace_world_entities: bool = False,
                 align_world_extractions: bool = False,
                 gold_world_extractions: bool = False,
                 tagger_only: bool = False,
                 denotation_only: bool = False,
                 world_extraction_model: Optional[str] = None,
                 skip_attributes_regex: Optional[str] = None,
                 entity_bits_mode: Optional[str] = None,
                 entity_types: Optional[List[str]] = None,
                 lexical_cues: List[str] = None,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._entity_token_indexers = self._question_token_indexers
        self._sample = sample
        self._replace_world_entities = replace_world_entities
        self._lf_syntax = lf_syntax
        self._entity_bits_mode = entity_bits_mode
        self._align_world_extractions = align_world_extractions
        self._gold_world_extractions = gold_world_extractions
        self._entity_types = entity_types
        self._tagger_only = tagger_only
        self._denotation_only = denotation_only
        self._skip_attributes_regex = None
        if skip_attributes_regex is not None:
            self._skip_attributes_regex = re.compile(skip_attributes_regex)
        self._lexical_cues = lexical_cues

        # Recording of entities in categories relevant for tagging
        all_entities = {}
        all_entities["world"] = ["world1", "world2"]
        # TODO: Clarify this into an appropriate parameter
        self._collapse_tags = ["world"]

        self._all_entities = None
        if entity_types is not None:
            if self._entity_bits_mode == "collapsed":
                self._all_entities = entity_types
            else:
                self._all_entities = [e for t in entity_types for e in all_entities[t]]

        logger.info(f"all_entities = {self._all_entities}")

        # Base world, depending on LF syntax only
        self._knowledge_graph = KnowledgeGraph(entities={"placeholder"}, neighbors={},
                                               entity_text={"placeholder": "placeholder"})
        self._world = QuarelWorld(self._knowledge_graph, self._lf_syntax)

        # Decide dynamic entities, if any
        self._dynamic_entities: Dict[str, str] = dict()
        self._use_attr_entities = False
        if "_attr_entities" in lf_syntax:
            self._use_attr_entities = True
            qr_coeff_sets = self._world.qr_coeff_sets
            for qset in qr_coeff_sets:
                for attribute in qset:
                    if (self._skip_attributes_regex is not None and
                                self._skip_attributes_regex.search(attribute)):
                        continue
                    # Get text associated with each entity, both from entity identifier and
                    # associated lexical cues, if any
                    entity_strings = [words_from_entity_string(attribute).lower()]
                    if self._lexical_cues is not None:
                        for key in self._lexical_cues:
                            if attribute in LEXICAL_CUES[key]:
                                entity_strings += LEXICAL_CUES[key][attribute]
                    self._dynamic_entities["a:"+attribute] = " ".join(entity_strings)

        # Update world to include dynamic entities
        if self._use_attr_entities:
            logger.info(f"dynamic_entities = {self._dynamic_entities}")
            neighbors: Dict[str, List[str]] = {key: [] for key in self._dynamic_entities}
            self._knowledge_graph = KnowledgeGraph(entities=set(self._dynamic_entities.keys()),
                                                   neighbors=neighbors,
                                                   entity_text=self._dynamic_entities)
            self._world = QuarelWorld(self._knowledge_graph, self._lf_syntax)

        self._stemmer = PorterStemmer().stemmer

        self._world_tagger_extractor = None
        self._extract_worlds = False
        if world_extraction_model is not None:
            logger.info("Loading world tagger model...")
            self._extract_worlds = True
            self._world_tagger_extractor = WorldTaggerExtractor(world_extraction_model)
            logger.info("Done loading world tagger model!")

        # Convenience regex for recognizing attributes
        self._attr_regex = re.compile(r"""\((\w+) (high|low|higher|lower)""")

    # Depending on dataset parameters, preprocess the data
    def preprocess(self, question_data: JsonDict, predict: bool = False) -> List[JsonDict]:
        # Use 'world_literals' to override 'world_extractions'
        if self._gold_world_extractions and 'world_literals' in question_data:
            question_data['world_extractions'] = question_data['world_literals']
        # Replace special substrings, like "(A)" and "___" to simplify tokenization later
        question_data['question'] = self._fix_question(question_data['question'])
        # Extract spans corresponding to 'worlds'
        if self._extract_worlds:
            if self._gold_world_extractions:
                logger.warning("Both gold_worlds and extract_worlds are set to True")
            # TODO: Keep token level information here?
            extractions = self._world_tagger_extractor.get_world_entities(question_data['question'])
            question_data['world_extractions'] = extractions
        if self._entity_types is not None:
            question_data['entity_literals'] = self._get_entity_literals(question_data)

        if 'logical_forms' in question_data:
            logical_forms = question_data['logical_forms']
            # If need be, updated gold LF with "a:" type prefix
            if self._use_attr_entities:
                logical_forms = [self._attr_regex.sub(r"(a:\1 \2", lf) for lf in logical_forms]
            # Check if the 1st or 2nd (flipped) logical form should be used
            if self._align_world_extractions and 'world_extractions' in question_data:
                world_flip = self._check_world_flip(question_data)
                if world_flip and len(logical_forms) > 1:
                    logical_forms = [logical_forms[1]]
                else:
                    logical_forms = [logical_forms[0]]
            question_data['logical_forms'] = logical_forms

        output = [question_data]
        need_extractions = self._replace_world_entities and not predict
        if not 'world_extractions' in question_data and need_extractions:
            output = []
        # Can potentially return different variants of question here, currently
        # output is either 0 or 1 entries
        if self._replace_world_entities and 'world_extractions' in question_data:
            output = [self._replace_stemmed_entities(data) for data in output]
        return output

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        # Set debug_counter to, say, 5 to get extra information logged for first 5 instances
        debug_counter = 5
        counter = self._sample
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                counter -= 1
                if counter == 0:
                    break
                line = line.strip("\n")
                if not line:
                    continue
                question_data_orig = json.loads(line)
                question_data_list = self.preprocess(question_data_orig)

                debug_counter -= 1
                if debug_counter > 0:
                    logger.info(f'question_data_list = {question_data_list}')
                for question_data in question_data_list:
                    question = question_data['question']
                    question_id = question_data['id']
                    logical_forms = question_data['logical_forms']
                    # Skip examples with certain attributes
                    if (self._skip_attributes_regex is not None and
                                self._skip_attributes_regex.search(logical_forms[0])):
                        continue
                    # Somewhat hacky filtering to "friction" subset of questions based on id
                    if not self._compatible_question(question_data):
                        continue

                    if debug_counter > 0:
                        logger.info(f'logical_forms = {logical_forms}')
                    answer_index = question_data['answer_index']
                    world_extractions = question_data.get('world_extractions')
                    entity_literals = question_data.get('entity_literals')
                    if entity_literals is not None and world_extractions is not None:
                        # This will catch flipped worlds if need be
                        entity_literals.update(world_extractions)
                    additional_metadata = {'id': question_id,
                                           'question': question,
                                           'answer_index': answer_index,
                                           'logical_forms': logical_forms}

                    yield self.text_to_instance(question, logical_forms,
                                                additional_metadata, world_extractions,
                                                entity_literals, debug_counter=debug_counter)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         logical_forms: List[str] = None,
                         additional_metadata: Dict[str, Any] = None,
                         world_extractions: Dict[str, Union[str, List[str]]] = None,
                         entity_literals: Dict[str, Union[str, List[str]]] = None,
                         tokenized_question: List[Token] = None,
                         debug_counter: int = None,
                         qr_spec_override: List[Dict[str, int]] = None,
                         dynamic_entities_override: Dict[str, str] = None) -> Instance:

        # pylint: disable=arguments-differ
        tokenized_question = tokenized_question or self._tokenizer.tokenize(question.lower())
        additional_metadata = additional_metadata or dict()
        additional_metadata['question_tokens'] = [token.text for token in tokenized_question]
        if world_extractions is not None:
            additional_metadata['world_extractions'] = world_extractions
        question_field = TextField(tokenized_question, self._question_token_indexers)

        if qr_spec_override is not None or dynamic_entities_override is not None:
            # Dynamically specify theory and/or entities
            dynamic_entities = dynamic_entities_override or self._dynamic_entities
            neighbors: Dict[str, List[str]] = {key: [] for key in dynamic_entities.keys()}
            knowledge_graph = KnowledgeGraph(entities=set(dynamic_entities.keys()),
                                             neighbors=neighbors,
                                             entity_text=dynamic_entities)
            world = QuarelWorld(knowledge_graph,
                                self._lf_syntax,
                                qr_coeff_sets=qr_spec_override)
        else:
            knowledge_graph = self._knowledge_graph
            world = self._world

        table_field = KnowledgeGraphField(knowledge_graph,
                                          tokenized_question,
                                          self._entity_token_indexers,
                                          tokenizer=self._tokenizer)

        if self._tagger_only:
            fields: Dict[str, Field] = {'tokens': question_field}
            if entity_literals is not None:
                entity_tags = self._get_entity_tags(self._all_entities, table_field,
                                                    entity_literals, tokenized_question)
                if debug_counter > 0:
                    logger.info(f'raw entity tags = {entity_tags}')
                entity_tags_bio = self._convert_tags_bio(entity_tags)
                fields['tags'] = SequenceLabelField(entity_tags_bio, question_field)
                additional_metadata['tags_gold'] = entity_tags_bio
            additional_metadata['words'] = [x.text for x in tokenized_question]
            fields['metadata'] = MetadataField(additional_metadata)
            return Instance(fields)

        world_field = MetadataField(world)

        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_actions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not world.is_table_entity(rule_right_side)
            field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        fields = {'question': question_field,
                  'table': table_field,
                  'world': world_field,
                  'actions': action_field}

        if self._denotation_only:
            denotation_field = LabelField(additional_metadata['answer_index'], skip_indexing=True)
            fields['denotation_target'] = denotation_field

        if self._entity_bits_mode is not None and world_extractions is not None:
            entity_bits = self._get_entity_tags(['world1', 'world2'], table_field,
                                                world_extractions, tokenized_question)
            if self._entity_bits_mode == "simple":
                entity_bits_v = [[[0, 0], [1, 0], [0, 1]][tag] for tag in entity_bits]
            elif self._entity_bits_mode == "simple_collapsed":
                entity_bits_v = [[[0], [1], [1]][tag] for tag in entity_bits]
            elif self._entity_bits_mode == "simple3":
                entity_bits_v = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]][tag] for tag in entity_bits]

            entity_bits_field = ArrayField(np.array(entity_bits_v))
            fields['entity_bits'] = entity_bits_field

        if logical_forms:
            action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore
            action_sequence_fields: List[Field] = []
            for logical_form in logical_forms:
                expression = world.parse_logical_form(logical_form)
                action_sequence = world.get_action_sequence(expression)
                try:
                    index_fields: List[Field] = []
                    for production_rule in action_sequence:
                        index_fields.append(IndexField(action_map[production_rule], action_field))
                    action_sequence_fields.append(ListField(index_fields))
                except KeyError as error:
                    logger.info(f'Missing production rule: {error.args}, skipping logical form')
                    logger.info(f'Question was: {question}')
                    logger.info(f'Logical form was: {logical_form}')
                    continue
            fields['target_action_sequences'] = ListField(action_sequence_fields)
        fields['metadata'] = MetadataField(additional_metadata or {})
        return Instance(fields)

    def _convert_tags_bio(self, tags: List[int]) -> List[str]:
        res = []
        last_tag = 0
        prefix_i = "I-"
        prefix_b = "B-"
        all_tags = self._all_entities
        if self._collapse_tags is not None:
            if 'world' in self._collapse_tags:
                all_tags = ['world' if 'world' in x else x for x in all_tags]
            if 'comparison' in self._collapse_tags:
                all_tags = ['comparison' if '-higher' in x or '-lower' in x else x for x in all_tags]
            if 'value' in self._collapse_tags:
                all_tags = ['value' if '-high' in x or '-low' in x else x for x in all_tags]
        if self._entity_bits_mode == "label":
            prefix_i = ""
            prefix_b = ""
        for tag in tags:
            if tag == 0:
                bio_tag = "O"
            elif tag == last_tag:
                bio_tag = prefix_i + all_tags[tag-1]
            else:
                bio_tag = prefix_b + all_tags[tag-1]
            last_tag = tag
            res.append(bio_tag)
        return res

    def _compatible_question(self, question_data: JsonDict) -> bool:
        question_id = question_data.get('id')
        if not question_id:
            return True
        if not '_friction' in self._lf_syntax:
            return True
        return '_Fr_' in question_id or 'Friction' in question_id

    @staticmethod
    def _fix_question(question: str) -> str:
        """
        Replace answer dividers (A), (B) etc with a unique token answeroptionA, answeroptionB, ...
        Replace '_____' with 'blankblank'
        """
        res = re.sub(r'\(([A-G])\)', r"answeroption\1", question)
        res = re.sub(r" *_{3,} *", " blankblank ", res)
        return res

    @staticmethod
    def _get_first(maybe_list: Any) -> Any:
        if not isinstance(maybe_list, list):
            return maybe_list
        elif not maybe_list:
            return None
        else:
            return maybe_list[0]

    def _check_world_flip(self, question_data: JsonDict) -> bool:
        if not 'world_literals' in question_data or not 'world_extractions' in question_data:
            return False
        flip = False
        world_extractions = question_data['world_extractions']
        extracted = [self._get_first(world_extractions[key]) for key in ['world1', 'world2']]
        literals = question_data['world_literals']
        aligned = align_entities(extracted, literals, self._stemmer)
        # If we haven't aligned two different things (including None), give up
        if len(set(aligned)) < 2:
            return flip
        aligned_dict = {key: value for key, value in zip(aligned, extracted)}
        extractions = {}
        for key in literals.keys():
            # if key is missing, then it must be assigned to None per above logic
            value = aligned_dict[key] if key in aligned_dict else aligned_dict[None]
            extractions[key] = value
        if extractions['world1'] != extracted[0]:
            flip = True
        return flip

    def _get_entity_tags(self,
                         entities: List[str],
                         table_field: KnowledgeGraphField,
                         entity_literals: JsonDict,
                         tokenized_question: List[Token]) -> List[int]:
        res = []
        # Hackily access last two feature extractors for table field (span overlaps which don't
        # depend on the actual table information)
        features = table_field._feature_extractors[8:]  # pylint: disable=protected-access
        for i, token in enumerate(tokenized_question):
            tag_best = 0
            score_max = 0.0
            for tag_index, tag in enumerate(entities):
                literals = entity_literals.get(tag, [])
                if not isinstance(literals, list):
                    literals = [literals]
                for literal in literals:
                    tag_tokens = self._tokenizer.tokenize(literal.lower())
                    scores = [fe(tag, tag_tokens, token, i, tokenized_question) for fe in features]
                    # Small tie breaker in favor of longer sequences
                    score = max(scores) + len(tag_tokens)/100
                    if score > score_max and score >= 0.5:
                        tag_best = tag_index + 1
                        score_max = score
            res.append(tag_best)
        return res

    # Flatten all relevant entity literals
    def _get_entity_literals(self, question_data: JsonDict) -> JsonDict:
        res: JsonDict = {}
        for key, value in question_data.items():
            if '_literals' in key and key.replace('_literals', '') in self._entity_types:
                res.update(value)
        return res

    def _stem_phrase(self, phrase: str) -> str:
        return re.sub(r"\w+", lambda x: self._stemmer.stem(x.group(0)), phrase)

    def _replace_stemmed_entities(self, question_data: JsonDict) -> JsonDict:
        entity_name_map = {"world1": "worldone", "world2":"worldtwo"}
        question = question_data['question']
        entities = question_data['world_extractions']
        entity_pairs: List[Tuple[str, str]] = []
        for key, value in entities.items():
            if not isinstance(value, list):
                entity_pairs.append((key, value))
            else:
                entity_pairs += [(key, v) for v in value]
        max_words = max([len(re.findall(r"\w+", string)) for _, string in entity_pairs])
        word_pos = [[match.start(0), match.end(0)] for match in re.finditer(r'\w+', question)]
        entities_stemmed = {self._stem_phrase(value): entity_name_map.get(key, key) for
                            key, value in entity_pairs}

        def substitute(string: str) -> str:
            replacement = entities_stemmed.get(self._stem_phrase(string))
            return replacement if replacement else string

        replacements = {}
        for num_words in range(1, max_words + 1):
            for i in range(len(word_pos) - num_words + 1):
                sub = question[word_pos[i][0]:word_pos[i+num_words-1][1]]
                new_sub = substitute(sub)
                if new_sub != sub:
                    replacements[re.escape(sub)] = new_sub

        if not replacements:
            return question_data

        pattern = "|".join(sorted(replacements.keys(), key=lambda x: -len(x)))
        regex = re.compile("\\b("+pattern+")\\b")
        res = regex.sub(lambda m: replacements[re.escape(m.group(0))], question)
        question_data['question'] = res
        return question_data
