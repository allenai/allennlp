# Miscellaneous helper functions for QuaRel parser.
#

from typing import Any, Dict, List, Set, Tuple, Union

import re

from nltk.metrics.distance import edit_distance
from nltk.stem import PorterStemmer as NltkPorterStemmer  # For typing

from allennlp.common.util import JsonDict
from allennlp.data.tokenizers import Token
from allennlp.semparse import util as semparse_util
from allennlp.semparse.worlds.quarel_world import QuarelWorld


#
# Temporary home for lexical cues associated with QuaRel attributes
#

LEXICAL_CUES: Dict[str, Dict[str, List[str]]] = {}

LEXICAL_CUES["synonyms"] = {
        "friction": ["resistance", "traction"],
        "speed": ["velocity", "pace"],
        "distance": ["length", "way"],
        "heat": ["temperature", "warmth", "smoke"],
        "smoothness": ["slickness", "roughness"],
        "acceleration": [],
        "amountSweat": ["sweat"],
        "apparentSize": ["size"],
        "breakability": ["brittleness"],
        "brightness": [],
        "exerciseIntensity": ["excercise"],
        "flexibility": [],
        "gravity": [],
        "loudness": [],
        "mass": ["weight"],
        "strength": ["power"],
        "thickness": [],
        "time": [],
        "weight": ["mass"]
}

LEXICAL_CUES["values"] = {
        "friction": [],
        "speed": ["fast", "slow", "faster", "slower", "slowly", "quickly", "rapidly"],
        "distance": ["far", "near", "further", "longer", "shorter", "long", "short",
                     "farther", "furthest"],
        "heat": ["hot", "hotter", "cold", "colder"],
        "smoothness": ["rough", "smooth", "rougher", "smoother", "bumpy", "slicker"],
        "acceleration": [],
        "amountSweat": ["sweaty"],
        "apparentSize": ["large", "small", "larger", "smaller"],
        "breakability": ["brittle", "break", "solid"],
        "brightness": ["bright", "shiny", "faint"],
        "exerciseIntensity": ["run", "walk"],
        "flexibility": ["flexible", "stiff", "rigid"],
        "gravity": [],
        "loudness": ["loud", "faint", "louder", "fainter"],
        "mass": ["heavy", "light", "heavier", "lighter", "massive"],
        "strength": ["strong", "weak", "stronger", "weaker"],
        "thickness": ["thick", "thin", "thicker", "thinner", "skinny"],
        "time": ["long", "short",],
        "weight": ["heavy", "light", "heavier", "lighter"]
}

#
#  Various utility functions
#


# Split entity names into words (camel case, hyphen or underscore)
RE_DECAMEL = re.compile(r"\B([A-Z])")
def words_from_entity_string(entity: str) -> str:
    res = entity.replace("_", " ").replace("-", " ")
    res = RE_DECAMEL.sub(r" \1", res)
    return res


def split_question(question: str) -> List[str]:
    return re.split(r' *\([A-F]\) *', question)


def nl_triple(triple: List[str], nl_world: JsonDict) -> str:
    return f"{nl_attr(triple[0]).capitalize()} is {triple[1]} for {nl_world[triple[2]]}"


def nl_arg(arg: Any, nl_world: JsonDict) -> Any:
    if arg[0] == 'and':
        return [nl_arg(x, nl_world) for x in arg[1:]]
    else:
        return [nl_triple(arg, nl_world)]


def nl_attr(attr: str) -> str:
    return words_from_entity_string(strip_entity_type(attr)).lower()


def nl_dir(sign: int) -> str:
    if sign == 1:
        return "higher"
    else:
        return "lower"


def nl_world_string(world: List[str]) -> str:
    return f'"{str_join(world, "|")}"'


def strip_entity_type(entity: str) -> str:
    return re.sub(r'^[a-z]:', '', entity)


def str_join(string_or_list: Union[str, List[str]],
             joiner: str,
             prefixes: str = "",
             postfixes: str = "") -> str:
    res = string_or_list
    if not isinstance(res, list):
        res = [res]
    res = [f'{prefixes}{x}{postfixes}' for x in res]
    return joiner.join(res)


def get_explanation(logical_form: str,
                    world_extractions: JsonDict,
                    answer_index: int,
                    world: QuarelWorld) -> List[JsonDict]:
    """
    Create explanation (as a list of header/content entries) for an answer
    """
    output = []
    nl_world = {}
    if world_extractions['world1'] != "N/A" and world_extractions['world1'] != ["N/A"]:
        nl_world['world1'] = nl_world_string(world_extractions['world1'])
        nl_world['world2'] = nl_world_string(world_extractions['world2'])
        output.append({
                "header": "Identified two worlds",
                "content": [f'''world1 = {nl_world['world1']}''',
                            f'''world2 = {nl_world['world2']}''']
        })
    else:
        nl_world['world1'] = 'world1'
        nl_world['world2'] = 'world2'
    parse = semparse_util.lisp_to_nested_expression(logical_form)
    if parse[0] != "infer":
        return None
    setup = parse[1]
    output.append({
            "header": "The question is stating",
            "content": nl_arg(setup, nl_world)
    })
    answers = parse[2:]
    output.append({
            "header": "The answer options are stating",
            "content": ["A: " + " and ".join(nl_arg(answers[0], nl_world)),
                        "B: " + " and ".join(nl_arg(answers[1], nl_world))]
    })
    setup_core = setup
    if setup[0] == 'and':
        setup_core = setup[1]
    s_attr = setup_core[0]
    s_dir = world.qr_size[setup_core[1]]
    s_world = nl_world[setup_core[2]]
    a_attr = answers[answer_index][0]
    qr_dir = world._get_qr_coeff(strip_entity_type(s_attr), strip_entity_type(a_attr))  # pylint: disable=protected-access
    a_dir = s_dir * qr_dir
    a_world = nl_world[answers[answer_index][2]]

    content = [f'When {nl_attr(s_attr)} is {nl_dir(s_dir)} ' +
               f'then {nl_attr(a_attr)} is {nl_dir(a_dir)} (for {s_world})']
    if a_world != s_world:
        content.append(f'''Therefore {nl_attr(a_attr)} is {nl_dir(-a_dir)} for {a_world}''')
    content.append(f"Therefore {chr(65+answer_index)} is the correct answer")

    output.append({
            "header": "Theory used",
            "content": content
    })

    return output


## Code for processing QR specs to/from string format

RE_GROUP = re.compile(r"\[([^[\]].*?)\]")
RE_SEP = re.compile(r" *, *")
RE_INITLETTER = re.compile(r" (.)")


def to_camel(string: str) -> str:
    return RE_INITLETTER.sub(lambda x: x.group(1).upper(), string)


def from_qr_spec_string(qr_spec: str) -> List[Dict[str, int]]:
    res = []
    groups = RE_GROUP.findall(qr_spec)
    for group in groups:
        group_split = RE_SEP.split(group)
        group_dict = {}
        for attribute in group_split:
            sign = 1
            if attribute[0] == "-":
                sign = -1
                attribute = attribute[1:]
            elif attribute[0] == "+":
                attribute = attribute[1:]
            attribute = to_camel(attribute)
            group_dict[attribute] = sign
        res.append(group_dict)
    return res


def to_qr_spec_string(qr_coeff_sets: List[Dict[str, int]]) -> str:
    res = []
    signs = {1:"+", -1:"-"}
    for qr_set in qr_coeff_sets:
        first = True
        group_list = []
        for attr, sign in qr_set.items():
            signed_attr = signs[sign] + attr
            if first:
                first = False
                if sign == 1:
                    signed_attr = attr
            group_list.append(signed_attr)
        res.append(f'[{", ".join(group_list)}]')
    return "\n".join(res)


def from_entity_cues_string(cues_string: str) -> Dict[str, List[str]]:
    lines = cues_string.split("\n")
    res = {}
    for line in lines:
        line_split = line.split(":")
        head = line_split[0].strip()
        cues: List[str] = []
        if len(line_split) > 1:
            cues = RE_SEP.split(line_split[1])
        res[head] = cues
    return res


def from_bio(tags: List[str], target: str) -> List[Tuple[int, int]]:
    res: List[Tuple[int, int]] = []
    current = None
    for index, tag in enumerate(tags):
        if tag == "B-" + target:
            if current is not None:
                res.append((current, index))
            current = index
        elif tag == "I-" + target:
            if current is None:
                current = index  # Should not happen
        else:
            if current is not None:
                res.append((current, index))
            current = None
    if current is not None:
        res.append((current, len(tags)))
    return res


def delete_duplicates(expr: List) -> List:
    seen: Set = set()
    res: List = []
    for expr1 in expr:
        if not expr1 in seen:
            seen.add(expr1)
            res.append(expr1)
    return res


def group_worlds(tags: List[str], tokens: List[str]) -> Dict[str, List[str]]:
    spans = from_bio(tags, 'world')
    with_strings = [(" ".join(tokens[i:j]), i, j) for i, j in spans]
    with_strings.sort(key=lambda x: len(x[0]), reverse=True)
    substring_groups: List[List[Tuple[str, int, int]]] = []
    ambiguous = []
    for string, i, j in with_strings:
        found = None
        for group_index, group in enumerate(substring_groups):
            for string_g, _, _ in group:
                if string in string_g:
                    if found is None:
                        found = group_index
                    elif found != group_index:
                        found = -1  # Found multiple times
        if found is None:
            substring_groups.append([(string, i, j)])
        elif found >= 0:
            substring_groups[found].append((string, i, j))
        else:
            ambiguous.append((string, i, j))
    nofit = []
    if len(substring_groups) > 2:
        substring_groups.sort(key=len, reverse=True)
        for extra in substring_groups[2:]:
            best_distance = 999
            best_index = None
            string = extra[0][0]  # Use the longest string
            for index, group in enumerate(substring_groups[:2]):
                for string_g, _, _ in group:
                    distance = edit_distance(string_g, string)
                    if distance < best_distance:
                        best_distance = distance
                        best_index = index
            # Heuristics for "close enough"
            if best_index is not None and best_distance < len(string) - 1:
                substring_groups[best_index] += extra
            else:
                nofit.append(extra)
    else:
        substring_groups += [[("N/A", 999, 999)]] * 2   # padding
    substring_groups = substring_groups[:2]
    # Sort by first occurrence
    substring_groups.sort(key=lambda x: min([y[1] for y in x]))
    world_dict = {}
    for index, group in enumerate(substring_groups):
        world_strings = delete_duplicates([x[0] for x in group])
        world_dict['world'+str(index+1)] = world_strings

    return world_dict


class WorldTaggerExtractor:

    def __init__(self, tagger_archive):
        from allennlp.models.archival import load_archive
        from allennlp.predictors import Predictor
        self._tagger_archive = load_archive(tagger_archive)
        self._tagger = Predictor.from_archive(self._tagger_archive)

    def get_world_entities(self,
                           question: str,
                           tokenized_question: List[Token] = None) -> Dict[str, List[str]]:

        # TODO: Fix protected access
        tokenized_question = tokenized_question or \
                             self._tagger._dataset_reader._tokenizer.tokenize(question.lower())  # pylint: disable=protected-access
        instance = self._tagger._dataset_reader.text_to_instance(question,  # pylint: disable=protected-access
                                                                 tokenized_question=tokenized_question)
        output = self._tagger._model.forward_on_instance(instance)  # pylint: disable=protected-access
        tokens_text = [t.text for t in tokenized_question]
        res = group_worlds(output['tags'], tokens_text)
        return res


def get_words(string: str) -> List[str]:
    return re.findall(r'[A-Za-z]+', string)


def get_stem_overlaps(query: str, references: List[str], stemmer: NltkPorterStemmer) -> List[int]:
    query_stems = {stemmer.stem(x) for x in get_words(query)}
    references_stems = [{stemmer.stem(x) for x in get_words(reference)} for reference in references]
    return [len(query_stems.intersection(reference_stems)) for reference_stems in references_stems]


def align_entities(extracted: List[str],
                   literals: JsonDict,
                   stemmer: NltkPorterStemmer) -> List[str]:
    """
    Use stemming to attempt alignment between extracted world and given world literals.
    If more words align to one world vs the other, it's considered aligned.
    """
    literal_keys = list(literals.keys())
    literal_values = list(literals.values())
    overlaps = [get_stem_overlaps(extract, literal_values, stemmer) for extract in extracted]
    worlds = []
    for overlap in overlaps:
        if overlap[0] > overlap[1]:
            worlds.append(literal_keys[0])
        elif overlap[0] < overlap[1]:
            worlds.append(literal_keys[1])
        else:
            worlds.append(None)
    return worlds
