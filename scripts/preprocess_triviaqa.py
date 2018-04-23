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
import json
import logging
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

from allennlp.common import Params
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.dataset_readers.reading_comprehension.triviaqa import process_triviaqa_questions

logger = logging.getLogger(__name__)

def main(triviaqa_path: pathlib.Path, outdir: pathlib.Path):
    outdir.mkdir(exist_ok=True)

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

    tokenizer = Tokenizer.from_params(Params({}))

    for questions_file in ['web-train.json', 'web-dev.json']:
        if questions_file == 'web-train.json':
            topn = 4
            require_answer = True
            instance_per_doc = True
        else:
            topn = 15
            require_answer = False
            instance_per_doc = False

        logger.info(f"starting questions file {questions_file}")
        questions_path = triviaqa_path / 'qa' / questions_file

        output_path = outdir / f"{questions_file}l"

        with open(output_path, 'w') as f:
            for question in process_triviaqa_questions(evidence_path=triviaqa_path,
                                                       questions_path=questions_path,
                                                       tokenizer=tokenizer,
                                                       max_paragraphs=topn,
                                                       instance_per_document=instance_per_doc,
                                                       require_answer=require_answer):
                f.write(json.dumps(question.to_json()))
                f.write("\n")

    # And then finally clean up:
    if tempdir is not None:
        logger.info(f"cleaning up tempdir {tempdir}")
        shutil.rmtree(tempdir)

if __name__ == '__main__':
    triviaqa_path = pathlib.Path(sys.argv[1])
    outdir = pathlib.Path(sys.argv[2])
    main(triviaqa_path, outdir)
