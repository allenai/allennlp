from collections import defaultdict, Counter
from typing import Dict, List
import logging
import json

import re
import numpy as np
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ProductionRuleField, MetadataField, KnowledgeGraphField
from allennlp.data.fields import Field, ListField, ArrayField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

# from java_programmer.fields import JavaProductionRuleField, JavaGlobalProductionRuleField, ProductionRuleField
from allennlp.semparse import KnowledgeGraph
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
IdentifierNT = 'IdentifierNT'
IdentifierOther = IdentifierNT + 'Other'
# LITERALS_TO_TRIM = [IdentifierNT, "Nt_decimal_literal", "Nt_char_literal",   "Nt_float_literal", "Nt_hex_literal", "Nt_oct_literal"]
IDENTIFIER_TYPES = ['Primary', 'ClassOrInterfaceType', 'Nt_33']
LITERALS_TO_TRIM = [IdentifierNT+IDENTIFIER_TYPES[0], IdentifierNT+IDENTIFIER_TYPES[1], IdentifierNT+IDENTIFIER_TYPES[2], IdentifierOther, "Nt_decimal_literal", "Nt_char_literal",   "Nt_float_literal", "Nt_hex_literal", "Nt_oct_literal"]
UNK = '<UNK>'
DUMMY = '<DUMMY>'
PLAIN_IDENTIFIER_RULE = "<SOME_NAME>-->" + "<SOME_NAME>"
STOPS = set(stopwords.words("english"))

@DatasetReader.register("java")
class JavaDatasetReader(DatasetReader):
    def __init__(self,
                 utterance_indexers: Dict[str, TokenIndexer],
                 type_indexers: Dict[str, TokenIndexer],
                 min_identifier_count: int,
                 num_dataset_instances: int,
                 tokenizer: Tokenizer = None,
                 linking_feature_extractors: List[str] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._utterance_indexers = utterance_indexers
        self._type_indexers = type_indexers

        self._environment_token_indexers = {"tokens": SingleIdTokenIndexer()}

        self._min_identifier_count = min_identifier_count
        self._num_dataset_instances = num_dataset_instances
        self._tokenizer = tokenizer or WordTokenizer()
        self._linking_feature_extractors = linking_feature_extractors

    # class ASTNode:
    #     def __init__(self, children, rule, desc):
    #         self.children = children
    #         self.rule = rule
    #         self.desc = desc
    #     def __str__(self):
    #         return self.rule
    # def is_nonterminal(self, nt):
    #     return nt[0].isupper()
    # def _is_terminal_rule(self, rule):
    #     lhs, rhs = rule.split('-->')
    #     return (
    #            "IdentifierNT" in lhs) \
    #            or re.match(r"^Nt_.*_literal-->.*", rule) \
    #            or rule == "<unk>"
    # def gen_ast(self, rules, i):
    #     curr = self.ASTNode([], rules[i], "")
    #     lhs, rhs = rules[i].split('-->')
    #     rhs_tokens = rhs.split('___')
    #     if self._is_terminal_rule(rules[i]):
    #         curr.desc += rhs
    #     else:
    #         for nt in rhs_tokens:
    #             if self.is_nonterminal(nt):
    #                 child, i = self.gen_ast(rules, i+1)
    #                 curr.children.append(child)
    #                 curr.desc += child.desc
    #             else:
    #                 curr.desc += nt
    #     return curr, i
    # def printMethodCaller(self, curr):
    #     lhs, rhs = curr.rule.split('-->')
    #     if rhs == 'Expression___.___Nt_33':
    #         print(curr.children[0].desc, curr.children[1].desc)
    #     for c in curr.children:
    #         self.printMethodCaller(c)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            # dataset = json.load(dataset_file)
            p2methods = json.load(dataset_file)

        dataset = []
        for _, methods in p2methods.items():
            dataset += methods
            if self._num_dataset_instances != -1:
                if len(dataset) > self._num_dataset_instances:
                    break
        # if self._num_dataset_instances != -1:
        #     dataset = dataset[:self._num_dataset_instances]
        # print("Dataset lenght", len(dataset))

        modified_rules = self.split_identifier_rule_into_multiple(
            [d['rules'] for d in dataset]
        )

        og_prototype_rules = [d['prototype_rules'] for d in dataset]
        prototype_rules = self.split_identifier_rule_into_multiple(
            [d['prototype_rules'] for d in dataset]
        )
        global_production_rule_fields, global_rule2index = self.get_global_rule_fields(modified_rules)

        for rules in prototype_rules:
            for i,rule in enumerate(rules):
                # lhs, rhs = rule.split('-->')
                # No class field/func identifier rules since these cannot be embedded
                if rule not in global_rule2index:
                    rules[i] = PLAIN_IDENTIFIER_RULE

        logger.info("Reading the dataset")
        for i, record in enumerate(dataset):
            instance = self.text_to_instance(record['nl'],
                                             record['varNames'],
                                             record['varTypes'],
                                             record['methodNames'],
                                             record['methodReturns'],
                                             global_production_rule_fields,
                                             global_rule2index,

                                             # Pass the modified identifier rules.
                                             rules=modified_rules[i],
                                             proto_rules=prototype_rules[i],
                                             proto_tokens=record['prototype_nl'],
                                             proto_code=record['prototype_code'],
                                             proto_og_rules=og_prototype_rules[i],
                                             protoMethodName=record['prototype_methodName'],
                                             methodName=record['methodName'],
                                             code=record['code'],
                                             path=record['path'])
            yield instance

    def get_types_used_frequently_even_if_not_in_class(self, rules, dataset):
        type_in_class_count = 0
        num_types = 0
        type2count = Counter()
        for i, d in enumerate(dataset):
            enviro_types = d['varTypes'] + d['methodReturns']
            for rule in rules[i]:
                lhs, rhs = rule.split('-->')
                if lhs == 'IdentifierNTClassOrInterfaceType':
                    inclass = False
                    for t in enviro_types:
                        if rhs in t:
                            type_in_class_count += 1
                            inclass = True
                            break
                    if not inclass:
                        type2count[rhs] += 1
                        # print('-'*20)
                        # print(enviro_types)
                        # print(rhs)
                    num_types += 1

        # print('Num type in class', type_in_class_count)
        # print('Num type total', num_types)
        # print(type2count.most_common(50))
        # print(sum([count for t, count in type2count.most_common(50)]))
        return [t for t, _ in type2count.most_common(50)]

    def experiment_nt33(self, rules, dataset):
        type_in_class_count = 0
        num_types = 0
        type2count = Counter()
        for i, d in enumerate(dataset):
            enviro_types = d['varTypes'] + d['methodReturns']
            for rule in rules[i]:
                lhs, rhs = rule.split('-->')
                if lhs == 'IdentifierNTNt_33':
                    print('-----------'*5)
                    for rule in rules[i]:
                        print(rule)
                    print(' '.join(d['code']))
                    print(lhs, rhs)


        exit()


    def split_identifier_rule_into_multiple(self, rules_lst):
        """ The identifier rule is split based on parent state prefix to cut down the
        search space"""

        IDENTIFIER_TYPES = ['Primary', 'ClassOrInterfaceType', 'Nt_33']
        new_rules = []
        for rules in rules_lst:
            stack = []
            new_rules.append([])
            for rule in rules:
                lhs, rhs = rule.split('-->')
                if IdentifierNT in rhs:
                    stack.append(lhs)
                    if lhs in IDENTIFIER_TYPES:
                        new_rules[-1].append(rule.replace(IdentifierNT, IdentifierNT + lhs))
                    else:
                        new_rules[-1].append(rule.replace(IdentifierNT, IdentifierOther))
                elif IdentifierNT in lhs:
                    nt = stack.pop()
                    if nt in IDENTIFIER_TYPES:
                        new_rules[-1].append(IdentifierNT+nt+'-->'+rhs)
                    else:
                        new_rules[-1].append(IdentifierOther + '-->' + rhs)
                else:
                    new_rules[-1].append(rule)

        return new_rules

    def get_global_rule_fields(self, rules):
        # Prepare global production rules to be used in JavaGrammarState.
        lhs2all_rhs = self.trim_grammar_identifiers_literals(rules)

        # Converts all the rules into ProductionRuleFields and returns a dictionary
        # from the rule to its index in the ListField of ProductionRuleFields.

        rule2index = {}
        # todo(rajas): look into why adding this first rule breaks code
        # Temporarily adding a padding action in first spot so that
        # 0 can be used as a padding action
        # production_rule_fields = [ProductionRuleField(rule='', is_global_rule=False)]
        # index = 1
        production_rule_fields = []
        index = 0
        for lhs, all_rhs in lhs2all_rhs.items():
            for rhs in all_rhs:
                rule = lhs + '-->' + rhs
                if 'sriniclass' in rule or 'srinifunc' in rule:
                    # Names from environment shouldn't be added to global rules.
                    continue

                # identifiers for class or interface types are removed, since these
                # should strictly be copied from the environment. 1/3 of cases though,
                # an identifiertype that needs to be used won't be in the environment.
                # however, unks should be part of the vocabulary.
                if (lhs == (IdentifierNT + 'ClassOrInterfaceType')) and (rhs != UNK and rhs != DUMMY):
                # if (lhs == (IdentifierNT + 'ClassOrInterfaceType')):
                    continue

                field = ProductionRuleField(rule, is_global_rule=True)
                production_rule_fields.append(field)

                rule2index[rule] = index
                index += 1

        # Add the trivial identifier rule which will be used for all prototype
        # identifiers and the prev action embedding for a rule after identifier
        field = ProductionRuleField(PLAIN_IDENTIFIER_RULE, is_global_rule=True)
        production_rule_fields.append(field)
        rule2index[PLAIN_IDENTIFIER_RULE] = index
        return production_rule_fields, rule2index

    @overrides
    def text_to_instance(self,  # type: ignore
                         utterance_tokens: List[str],
                         variable_names: List[str],
                         variable_types: List[str],
                         method_names: List[str],
                         method_types: List[str],
                         global_rule_fields: List[ProductionRuleField],
                         globalrule2index: Dict[str, int],
                         code: str = None,
                         rules: List[str] = None,
                         proto_rules: List[str] = None,
                         proto_tokens: List[str] = None,
                         proto_code: List[str] = None,
                         proto_og_rules:List[str] = None,
                         protoMethodName: str = None,
                         methodName: str = None,
                         path: str = None
                         ) -> Instance:
        fields = {}

        # variable_name_fields = self.get_field_from_method_variable_names(variable_names)
        # method_name_fields = self.get_field_from_method_variable_names(method_names)
        # variable_types_field = TextField([Token(t.lower()) for t in variable_types], self._type_indexers)
        # method_return_types_field = TextField([Token(t.lower()) for t in method_types], self._type_indexers)


        # todo(rajas) add camel casing for utterance tokens as well
        # utterance = [t for word in utterance for t in self.split_camel_case_add_original(word)]

        # toks = ['<CURR>'] + utterance[:10]
        # toks += ['<PROTOTYPE>'] + proto_tokens[:10]
        # toks = self.split_camel_case(methodName) + ['<SEP>'] + utterance[:25]
        utterance_tokens = [t for t in utterance_tokens if t not in STOPS]
        utterance_tokens = utterance_tokens[:20]
        utterance_tokens = [t.lower() for t in utterance_tokens]
        utterance_tokens_type = [Token(t) for t in utterance_tokens]
        utterance_field = TextField(utterance_tokens_type,
                                    self._utterance_indexers)


        # proto_ = self.split_camel_case(protoMethodName) +['<SEP>'] + proto_tokens[:25]
        proto_tokens = [t for t in proto_tokens if t not in STOPS]
        proto_tokens = proto_tokens[:20]
        proto_tokens = [t.lower() for t in proto_tokens]
        proto_utterance_field = TextField([Token(t) for t in proto_tokens],
                                          self._utterance_indexers)

        metadata_dict = {'utterance':  utterance_tokens,
                         'prototype_utterance': proto_tokens,
                         # 'variableNames':  variable_names,
                         # 'variableTypes':  variable_types,
                         # 'methodNames':  method_names,
                         # 'methodTypes':  method_types,
                         'prototype_methodName': protoMethodName,
                         'methodName': methodName,
                         'path': path}
        if code is not None:
            metadata_dict['code'] = code
            metadata_dict['prototype_code'] = proto_code
        fields['metadata'] = MetadataField(metadata_dict)

        knowledge_graph, entity2isType = self.get_java_class_knowledge_graph(variable_names=variable_names,
                                                              variable_types=variable_types,
                                                              method_names=method_names,
                                                              method_types=method_types,
                                                              proto_rules=proto_og_rules)

        # We need to iterate over knowledge_graph's entities since these are sorted and each entity in
        # entity_tokens needs to correspond to knowledge_graph's entities.
        entity_tokens = []
        for entity in knowledge_graph.entities:
            entity_text = knowledge_graph.entity_text[entity]
            # Entity tokens should contain the camel case split of entity
            # e.g. isParsed -> is, Parsed and isParsed
            # So if the utterance has word parsed, it will appear in the exact text set
            # of knowledge graph field.
            entity_camel_split = self.split_camel_case_add_original(entity_text)
            entity_tokens.append([Token(e) for e in entity_camel_split])

        # todo(rajas): add a feature that filters out trivial links between utterance
        # and variables when the utterance has word 'is' and variable is 'isParsed'.
        # however if utterance has 'parsed' then it should be linked with 'isParsed'.
        java_class_field = KnowledgeGraphField(knowledge_graph=knowledge_graph,
                                               utterance_tokens=utterance_tokens_type,
                                               token_indexers=self._environment_token_indexers,
                                               entity_tokens=entity_tokens,
                                               feature_extractors=self._linking_feature_extractors)

        environment_rules, environmentrule2index = self.get_java_class_specific_rules(knowledge_graph, entity2isType)

        if rules is not None:
            target_rule_field = self.get_target_rule_field(rules, globalrule2index, environmentrule2index)
            fields['rules'] = target_rule_field

        if proto_rules is not None:
            # todo(rajas) just make this a production rule field
            proto_rule_field = self.get_target_rule_field(proto_rules, globalrule2index, environmentrule2index)
            fields['prototype_rules'] = proto_rule_field

        entity_field = MetadataField(knowledge_graph.entities)

        fields.update({"utterance": utterance_field,
                       "prototype_utterance": proto_utterance_field,
                       # "variable_names": variable_name_fields,
                       # "variable_types": variable_types_field,
                       # "method_names": method_name_fields,
                       # "method_return_types": method_return_types_field,
                       "actions": ListField(global_rule_fields + environment_rules),
                       "java_class": java_class_field,
                       "entities": entity_field })

        return Instance(fields)



    def trim_grammar_identifiers_literals(self, rules_list):
        # This is used remove identifier and literal rules, if they occur below
        # a self._min_identifier_count. This is needed since the action_indexer
        # indexes the entire rule, so instead of having it map the entire rule
        # to an unk we want the rule to become Identifier-->UNK

        # First get all unique rhsides and their counts.
        lhs2rhscounts = defaultdict(Counter)
        for rules in rules_list:
            for rule in rules:
                lhs, rhs = rule.split('-->')
                lhs2rhscounts[lhs][rhs] += 1

        lhs2trimmedrhs = defaultdict(list)
        for lhs in lhs2rhscounts:
            # We only trim from a predefined set of identifiers and literals. This
            # means that some literals aren't trimmed, but that's because the
            # expand to a small set of terminals so it's not necessary.
            if lhs in LITERALS_TO_TRIM:
                for rhs, count in lhs2rhscounts[lhs].items():
                    if count >= self._min_identifier_count:
                        lhs2trimmedrhs[lhs].append(rhs)
            else:
                for rhs, _ in lhs2rhscounts[lhs].items():
                    lhs2trimmedrhs[lhs].append(rhs)

        # Now add the unk rules for each literal
        for nt in LITERALS_TO_TRIM:
            # if nt in lhs2trimmedrhs:
            # We could check first whether any trimming occurred, but this
            # is guaranteed to happen so we add unk by default.
            lhs2trimmedrhs[nt].append(UNK)
            # A DUMMY action is added for when UNK's are removed during test time
            # to prevent the number of embedded actions from being 0.
            # lhs2trimmedrhs[nt].append(DUMMY)

        return lhs2trimmedrhs


    def get_java_class_knowledge_graph(self,
                                       variable_names,
                                       variable_types,
                                       method_names,
                                       method_types,
                                       proto_rules=None):
        # # Note that sriniclass_ prefix means that the name is copied from the
        # # java class.

        # todo(rajas): handle types like List<String>. ideally this should
        # map to 3 types


        entities = []
        entity_text = {}
        neighbors = defaultdict(list)
        entity2isType = {}
        for name, type in zip(variable_names, variable_types):
            entity = 'sriniclass_' + name
            entities.append(entity)
            entity_text[entity] = name
            entity2isType[entity] = False

            split_types = self.split_type(type)
            for t in split_types:
                entity2isType[t] = True

                if t not in entities:
                    # We don't want to add the same type such as "Object" repeatedly.
                    entities.append(t)
                    entity_text[t] = t
                neighbors[entity].append(t)
                neighbors[t].append(entity)
        for name, type in zip(method_names, method_types):
            entity = 'srinifunc_' + name
            entities.append(entity)
            entity_text[entity] = name
            entity2isType[entity] = False

            split_types = self.split_type(type)
            for t in split_types:
                entity2isType[t] = True

                if t not in entities:
                    # We don't want to add the same type such as "Object" repeatedly.
                    entities.append(t)
                    entity_text[t] = t
                neighbors[entity].append(t)
                neighbors[t].append(entity)

        # todo(rajas) this can get sped up by iterating over rules instead
        entity2in_prototype = {}
        for e in entities:
            for rule in proto_rules:
                if e in rule:
                    entity2in_prototype[e] = True


        # neighbors = {entity: [] for entity in entity_text}
        knowledge_graph = KnowledgeGraph(entities=entities,
                                         entity_text=entity_text,
                                         neighbors=neighbors,
                                         entity2in_prototype=entity2in_prototype)
        return knowledge_graph, entity2isType

    def split_type(self, type):
        # Split List<String> into [List, String]
        types = re.split('[^a-zA-Z0-9]', type)
        return [t for t in types if t != '']

    def get_java_class_specific_rules(self, knowledge_graph: KnowledgeGraph, entity2isType):
        # todo(rajas): clean this method up, ideally entity2isType should be part of the knowledge graph.

        # Returns the rules unique per training instance, which copy variable
        # and method names from the java class.
        java_class_rules = []
        javaclassrule2index = {}
        index = 0
        for entity in knowledge_graph.entities:
            # Variable names and method names are only generated from the IdentifierNT rule which
            # was generated from Primary, Nt_33, and two others. The other two occur less than 10
            # times so they are omitted.

            if entity2isType[entity] == False:
                rule1 = IdentifierNT + 'Primary-->' + entity
                rule2 = IdentifierNT + 'Nt_33-->' + entity

                field1 = ProductionRuleField(rule1, is_global_rule=False)
                field2 = ProductionRuleField(rule2, is_global_rule=False)
                java_class_rules.extend([field1, field2])

                javaclassrule2index[rule1] = index
                javaclassrule2index[rule2] = index + 1
                index += 2
            else:
                # Since entity type is a type, make rule class or interface type
                rule1 = IdentifierNT + 'ClassOrInterfaceType-->' + entity

                field1 = ProductionRuleField(rule1, is_global_rule=False)
                java_class_rules.append(field1)

                javaclassrule2index[rule1] = index
                index += 1
        return java_class_rules, javaclassrule2index

    def get_target_rule_field(self, rules, globalrule2index, environmentrule2index):
        rule_indexes = []
        for rule in rules:
            # It's important that the envirorule if statement comes first. In the case
            # that there is rule for the type from the environment, we want to encourage
            # the parser to copy from the environment.
            if rule in environmentrule2index:
                # Environment rule indexes come after last global index since the
                # environment rules are appended to the global rules list.
                rule_indexes.append(len(globalrule2index) + environmentrule2index[rule])
            elif rule in globalrule2index:
                rule_indexes.append(globalrule2index[rule])
            else:
                lhs, _ = rule.split('-->')
                rule_indexes.append(globalrule2index[lhs + '-->' + UNK])

        # todo(rajas) convert to an index field
        rule_field = ArrayField(np.array(rule_indexes), padding_value=-1)
        return rule_field

    def get_field_from_method_variable_names(self, words: List[str]) -> ListField:
        # For each variable or method name, this method splits it
        # on camel case then generates a TextField for each one.
        fields: List[Field] = []
        for word in words:
            tokens = [Token(text=w) for w in self.split_camel_case_add_original(word)]
            fields.append(TextField(tokens, self._utterance_indexers))
        return ListField(fields)

    def split_camel_case_add_original(self, name: str) -> List[str]:
        # Returns the string and its camel case split version.
        tokens = self.split_camel_case(name)
        if len(tokens) > 1:
            tokens = [name] + tokens
        return tokens

    @staticmethod
    def split_camel_case(name: str) -> List[str]:
        # Returns the string and its camel case split version.
        tokens = re.sub('(?!^)([A-Z][a-z]+)', r' \1', name).split()
        return tokens

    @classmethod
    def from_params(cls, params: Params) -> 'JavaDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        # todo(rajas): utterance indexer should be renamed to identifier indexer
        utterance_indexers = TokenIndexer.dict_from_params(params.pop('utterance_indexers'))
        min_identifier_count = params.pop_int('min_identifier_count')
        num_dataset_instances = params.pop_int('num_dataset_instances', -1)
        linking_feature_extracters = params.pop('linking_feature_extractors', None)
        # identifier_indexers = TokenIndexer.dict_from_params(params.pop('identifier_indexers'))
        type_indexers = TokenIndexer.dict_from_params(params.pop('type_indexers'))
        params.assert_empty(cls.__name__)
        return cls(utterance_indexers=utterance_indexers,
                   min_identifier_count=min_identifier_count,
                   num_dataset_instances=num_dataset_instances,
                   linking_feature_extractors=linking_feature_extracters,
                   # identifier_indexers=identifier_indexers,
                   type_indexers=type_indexers,
                   tokenizer=tokenizer,
                   lazy=lazy)