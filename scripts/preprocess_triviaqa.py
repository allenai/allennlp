"""
Takes the raw `triviaqa-rc.tar.gz` (or an untarred version of it)
and converts it into two JSONL files: `web-train.jsonl` and `web-dev.jsonl`.

Each JSON line corresponds to a single question and has the format

{
    "id": "qw_1934",
    "text": "what is the ... ?",
    "tokens": [["what", 0], ...],
    "paragraphs": {
        "text": ["first paragraph...", "second paragraph", ...],
        "tokens": [[["first", 0], ...], ...],
        "token_spans": [[[62, 63]], ...],
        "has_answers": [0],
    },
    "answer_texts": ["primary answer", "alternative answer"]
}
"""

from typing import List
import json
import logging
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

from allennlp.common import Params, JsonDict
from allennlp.common.util import lazy_groups_of
from allennlp.data import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.token import Token, token_to_json, json_to_token
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.dataset_readers.reading_comprehension.triviaqa import _PARAGRAPH_TOKEN
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

tfidf = TfidfVectorizer(strip_accents="unicode", stop_words="")
tokenizer: Tokenizer = None

def document_tfidf(paragraphs: List[str], question: str) -> np.ndarray:
    try:
        # (num_paragraphs, num_features)
        para_features = tfidf.fit_transform(paragraphs)
        # (1, num_features)
        q_features = tfidf.transform([question])
    except ValueError:
        # (num_paragraphs,)
        return np.array([0.0] * len(paragraphs))
    # pairwise_distances is (1, num_paragraphs), after ravel is (num_paragraphs,)
    dists = pairwise_distances(q_features, para_features, "cosine").ravel()
    return dists

def merge_and_sort(paragraphs: List[str],
                   question: str,
                   answer_texts: List[str],
                   topn: int = 4,
                   max_size: int = 400) -> JsonDict:
        tokens = []
        for paragraph in paragraphs:
            tokens.extend(token.text for token in tokenizer.tokenize(paragraph))
            tokens.append(_PARAGRAPH_TOKEN)

        # Get rid of trailing paragraph token
        tokens = tokens[:-1]

        merged_paragraphs = [' '.join(paragraph_tokens) for paragraph_tokens in lazy_groups_of(iter(tokens), max_size)]
        merged_paragraph_tokens = [tokenizer.tokenize(paragraph) for paragraph in merged_paragraphs]

        # Sort the paragraphs by their tfidf score with the question.
        scores = document_tfidf(merged_paragraphs, question)
        # Get the ranked indexes.
        ranks = [i for i, _ in sorted(enumerate(scores), key=lambda pair: pair[1])]

        # Find the indexes of paragraphs that have answers.
        has_answers = [i for i in ranks
                       if util.find_valid_answer_spans(merged_paragraph_tokens[i], answer_texts)]

        if not has_answers:
            return {}

        first_answer = has_answers[0]
        # Want first_answer to be the first paragraph, and then take the most highly ranked
        # other topn - 1
        choices = [first_answer] + [i for i in ranks if i != first_answer][:(topn - 1)]

        texts = [merged_paragraphs[i] for i in choices]
        tokens = [merged_paragraph_tokens[i] for i in choices]
        token_spans = [util.find_valid_answer_spans(tokens_i, answer_texts)
                       for tokens_i in tokens]

        return {
            "texts": texts,
            "tokens": [
                [token_to_json(token) for token in tokens_i]
                for tokens_i in tokens
            ],
            "token_spans": token_spans,
            "has_answers": [i for i, choice in enumerate(choices) if choice in has_answers]
        }

def main(params: Params, triviaqa_path: pathlib.Path, outdir: pathlib.Path):
    global tokenizer

    outdir.mkdir(exist_ok=True)

    result_key, evidence_subdir = 'SearchResults', 'web'
    # result_key, evidence_subdir = 'EntityPages', 'wikipedia'

    # If triviaqa is a tar.gz, then untar it to a temporary location:
    if triviaqa_path.is_dir():
        # Given a directory, so do nothing.
        logger.info(f"{triviaqa_path} is a directory, so no un-tar-ing to do.")
        tempdir = None
    else:
        # Make a new tempdir, untar the dataset, and store the location.
        tempdir = tempfile.mkdtemp()
        logger.info(f"Un-tar-ing {triviaqa_path} to {tempdir}")
        with tarfile.open(triviaqa_path) as tarball:
            tarball.extractall(tempdir)
        triviaqa_path = pathlib.Path(tempdir)

    tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))

    for questions_file, topn in [('web-train.json', 4),
                                 ('web-dev.json', 15)]:
        logger.info(f"starting questions file {questions_file}")
        questions_path = triviaqa_path / 'qa' / questions_file
        output_path = outdir / f"{questions_file}l"
        with open(questions_path, 'r') as f:
            questions_data = json.loads(f.read())['Data']

        with open(output_path, 'w') as f:
            for i, question in enumerate(questions_data):
                question_id = question['QuestionId']
                question_text = question['Question']
                logger.info(f"{i} {question_id} {question_text}")
                question_tokens = tokenizer.tokenize(question_text)

                answer = question['Answer']
                human_answers = [util.normalize_text(human_answer)
                                 for human_answer in answer.get('HumanAnswers', [])]
                answer_texts = answer['NormalizedAliases'] + human_answers
                evidence_files = [result['Filename'] for result in question[result_key]]

                paragraphs: List[str] = []

                for evidence_file in evidence_files:
                    evidence_path = triviaqa_path / 'evidence' / evidence_subdir / evidence_file
                    with open(evidence_path, 'r') as evidence_file:
                        paragraphs.extend(evidence_file.readlines())

                merged_paragraphs = merge_and_sort(paragraphs, question_text, answer_texts, topn)

                if not merged_paragraphs:
                    logger.warning(f"found no paragraphs with answers for {question_id}, skipping")
                    continue

                question_json = {
                    "id": question_id,
                    "text": question_text,
                    "tokens": [token_to_json(token) for token in question_tokens],
                    "paragraphs": merged_paragraphs,
                    "answer_texts": answer_texts
                }

                f.write(json.dumps(question_json))
                f.write("\n")

    # And then finally clean up:
    if tempdir is not None:
        logger.info(f"cleaning up tempdir {tempdir}")
        shutil.rmtree(tempdir)

if __name__ == '__main__':
    params = Params.from_file(sys.argv[1])
    triviaqa_path = pathlib.Path(sys.argv[2])
    outdir = pathlib.Path(sys.argv[3])
    main(params, triviaqa_path, outdir)
