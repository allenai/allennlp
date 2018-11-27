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
         out_fn: str) -> None:
    """
    inp_fn: str, required.
       Path to file from which to read CoNLL with space delimited fields.
    out_fn: str, required.
       Path to file to which to write the CoNLL format Open IE extractions.
    """
    with open(out_fn, 'w') as fout:
        for line in open(inp_fn):
            if line.startswith("#") or not(line.strip()):
                fout.write(line)
            else:
                fout.write("\t".join(line.strip().split()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Open IE4 extractions to CoNLL (ontonotes) format.")
    parser.add_argument("--inp", type=str, help="input file from which to read CoNLL with space delimited fields.", required = True)
    parser.add_argument("--out", type=str, help="path to the output file, where CoNLL format should be written.", required = True)
    args = parser.parse_args()
    main(args.inp, args.out)

