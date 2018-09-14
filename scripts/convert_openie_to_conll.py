import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from typing import List
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from collections import namedtuple
import regex
from tqdm import tqdm
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import WordTokenizer
import argparse

Extraction = namedtuple("Extraction",  # Open IE extraction
                        ["sent",       # Sentence in which this extraction appears
                         "toks",       # spaCy toks
                         "arg1",       # Subject
                         "rel",        # Relation
                         "args2",      # A list of arguments after the predicate
                         "confidence"] # Confidence in this extraction
)

Element = namedtuple("Element",    # An element (predicate or argument) in an Open IE extraction
                     ["elem_type", # Predicate or argument ID
                      "span",      # The element's character span in the sentence
                      "text"]      # The textual representation of this element
)

def main(inp_fn: str,
         domain: str,
         out_fn: str) -> None:
    """
    inp_fn: str, required.
       Path to file from which to read Open IE extractions in Open IE4's format.
    domain: str, required.
       Domain to be used when writing CoNLL format.
    out_fn: str, required.
       Path to file to which to write the CoNLL format Open IE extractions.
    """
    with open(out_fn, 'w') as fout:
        for sent_ls in read(inp_fn):
            fout.write("{}\n\n".format('\n'.join(['\t'.join(map(str,
                                                                pad_line_to_ontonotes(line,
                                                                                      domain)))
                                                  for line
                                                  in convert_sent_to_conll(sent_ls)])))

def safe_zip(*args):
    """
    Zip which ensures all lists are of same size.
    """
    assert (len(set(map(len, args))) == 1)
    return zip(*args)

def char_to_word_index(char_ind: int,
                       sent: str) -> int:
    """
    Convert a character index to
    word index in the given sentence.
    """
    return sent[: char_ind].count(' ')

def element_from_span(span: List[int],
                      span_type: str) -> Element:
    """
    Return an Element from span (list of spacy toks)
    """
    return Element(span_type,
                   [span[0].idx,
                    span[-1].idx + len(span[-1])],
                   ' '.join(map(str, span)))

def split_predicate(ex: Extraction) -> Extraction:
    """
    Ensure single word predicate
    by adding "before-predicate" and "after-predicate"
    arguments.
    """
    rel_toks = ex.toks[char_to_word_index(ex.rel.span[0], ex.sent) \
                       : char_to_word_index(ex.rel.span[1], ex.sent) + 1]
    if not rel_toks:
        return ex

    verb_inds = [tok_ind for (tok_ind, tok)
                 in enumerate(rel_toks)
                 if tok.tag_.startswith('VB')]

    last_verb_ind = verb_inds[-1] if verb_inds \
                    else (len(rel_toks) - 1)

    rel_parts = [element_from_span([rel_toks[last_verb_ind]],
                                   'V')]

    before_verb = rel_toks[ : last_verb_ind]
    after_verb = rel_toks[last_verb_ind + 1 : ]

    if before_verb:
        rel_parts.append(element_from_span(before_verb, "BV"))

    if after_verb:
        rel_parts.append(element_from_span(after_verb, "AV"))

    return Extraction(ex.sent, ex.toks, ex.arg1, rel_parts, ex.args2, ex.confidence)

def extraction_to_conll(ex: Extraction) -> List[str]:
    """
    Return a conll representation of a given input Extraction.
    """
    ex = split_predicate(ex)
    toks = ex.sent.split(' ')
    ret = ['*'] * len(toks)
    args = [ex.arg1] + ex.args2
    rels_and_args = [("ARG{}".format(arg_ind), arg)
                     for arg_ind, arg in enumerate(args)] + \
                         [(rel_part.elem_type, rel_part)
                          for rel_part
                          in ex.rel]

    for rel, arg in rels_and_args:
        # Add brackets
        cur_start_ind = char_to_word_index(arg.span[0],
                                           ex.sent)
        cur_end_ind = char_to_word_index(arg.span[1],
                                         ex.sent)
        ret[cur_start_ind] = "({}{}".format(rel, ret[cur_start_ind])
        ret[cur_end_ind] += ')'
    return ret

def interpret_span(text_spans: str) -> List[int]:
    """
    Return an integer tuple from
    textual representation of closed / open spans.
    """
    m = regex.match("^(?:(?:([\(\[]\d+, \d+[\)\]])|({\d+}))[,]?\s*)+$",
                    text_spans)

    spans = m.captures(1) + m.captures(2)

    int_spans = []
    for span in spans:
        ints = list(map(int,
                        span[1: -1].split(',')))
        if span[0] == '(':
            ints[0] += 1
        if span[-1] == ']':
            ints[1] += 1
        if span.startswith('{'):
            assert(len(ints) == 1)
            ints.append(ints[0] + 1)

        assert(len(ints) == 2)

        int_spans.append(ints)

    # Merge consecutive spans
    ret = []
    cur_span = int_spans[0]
    for (start, end) in int_spans[1:]:
        if start - 1 == cur_span[-1]:
            cur_span = (cur_span[0],
                        end)
        else:
            ret.append(cur_span)
            cur_span = (start, end)

    if (not ret) or (cur_span != ret[-1]):
        ret.append(cur_span)

    return ret[0]

def interpret_element(element_type: str, text: str, span: str) -> Element:
    """
    Construct an Element instance from regexp
    groups.
    """
    return Element(element_type,
                   interpret_span(span),
                   text)

def parse_element(raw_element: str) -> List[Element]:
    """
    Parse a raw element into text and indices (integers).
    """
    elements = [regex.match("^(([a-zA-Z]+)\(([^;]+),List\(([^;]*)\)\))$",
                            elem.lstrip().rstrip())
                for elem
                in raw_element.split(';')]
    return [interpret_element(*elem.groups()[1:])
            for elem in elements
            if elem]


def read(fn: str) -> List[Extraction]:
    tokenizer = WordTokenizer(word_splitter = SpacyWordSplitter(pos_tags=True))
    prev_sent = []

    with open(fn) as fin:
        for line in tqdm(fin):
            data = line.strip().split('\t')
            confidence = data[0]
            if not all(data[2:5]):
                # Make sure that all required elements are present
                continue
            arg1, rel, args2 = map(parse_element,
                                   data[2:5])

            # Exactly one subject and one relation
            # and at least one object
            if ((len(rel) == 1) and \
                (len(arg1) == 1) and \
                (len(args2) >= 1)):
                sent = data[5]
                cur_ex = Extraction(sent = sent,
                                    toks = tokenizer.tokenize(sent),
                                    arg1 = arg1[0],
                                    rel = rel[0],
                                    args2 = args2,
                                    confidence = confidence)


                # Decide whether to append or yield
                if (not prev_sent) or (prev_sent[0].sent == sent):
                    prev_sent.append(cur_ex)
                else:
                    yield prev_sent
                    prev_sent = [cur_ex]
    if prev_sent:
        # Yield last element
        yield prev_sent

def convert_sent_to_conll(sent_ls: List[Extraction]):
    """
    Given a list of extractions for a single sentence -
    convert it to conll representation.
    """
    # Sanity check - make sure all extractions are on the same sentence
    assert(len(set([ex.sent for ex in sent_ls])) == 1)
    toks = sent_ls[0].sent.split(' ')

    return safe_zip(*[range(len(toks)),
                      toks] + \
                    [extraction_to_conll(ex)
                     for ex in sent_ls])


def pad_line_to_ontonotes(line, domain) -> List[str]:
    """
    Pad line to conform to ontonotes representation.
    """
    word_ind, word = line[ : 2]
    pos = 'XX'
    oie_tags = line[2 : ]
    line_num = 0
    parse = "-"
    lemma = "-"
    return [domain, line_num, word_ind, word, pos, parse, lemma, '-',\
            '-', '-', '*'] + list(oie_tags) + ['-', ]

def convert_sent_dict_to_conll(sent_dic, domain) -> str:
    """
    Given a dictionary from sentence -> extractions,
    return a corresponding CoNLL representation.
    """
    return '\n\n'.join(['\n'.join(['\t'.join(map(str, pad_line_to_ontonotes(line, domain)))
                                   for line in convert_sent_to_conll(sent_ls)])
                        for sent_ls
                        in sent_dic.iteritems()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Open IE4 extractions to CoNLL (ontonotes) format.")
    parser.add_argument("--inp", type=str, help="input file from which to read Open IE extractions.", required = True)
    parser.add_argument("--domain", type=str, help="domain to use when writing the ontonotes file.", required = True)
    parser.add_argument("--out", type=str, help="path to the output file, where CoNLL format should be written.", required = True)
    args = parser.parse_args()
    main(args.inp, args.domain, args.out)

