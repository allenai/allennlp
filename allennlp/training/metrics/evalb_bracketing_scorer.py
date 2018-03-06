
from typing import List
import os
import tempfile
import subprocess

from overrides import overrides
from nltk import Tree

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("evalb")
class EvalbBracketingScorer(Metric):
    """
    This class uses the external EVALB software for computing a broad range of metrics
    on parse trees. Here, we use it to compute the Precision, Recall and F1 metrics.
    You can download the source for EVALB from here: <http://nlp.cs.nyu.edu/evalb/>.

    Note that this software is 20 years old. In order to compile it on modern hardware,
    you may need to remove an ``include <malloc.h>`` statement in ``evalb.c`` before it
    will compile.

    AllenNLP contains the EVALB software, but you will need to compile it yourself
    before using it because the binary it generates is system depenedent. To build it,
    run ``make`` inside the ``scripts/EVALB`` directory.

    Note that this metric reads and writes from disk quite a bit. You probably don't
    want to include it in your training loop; instead, you should calculate this on
    a validation set only.

    Parameters
    ----------
    evalb_directory_path : ``str``, required.
        The directory containing the EVALB executable.
    evalb_param_filename: ``str``, optional (default = "COLLINS.prm")
        The relative name of the EVALB configuration file used when scoring the trees.
        By default, this uses the COLLINS.prm configuration file which comes with EVALB.
        This configuration ignores POS tags and some punctuation labels.
    """
    def __init__(self, evalb_directory_path: str, evalb_param_filename: str = "COLLINS.prm") -> None:
        self._evalb_program_path = os.path.join(evalb_directory_path, "evalb")
        self._evalb_param_path = os.path.join(evalb_directory_path, evalb_param_filename)

        if not os.path.exists(self._evalb_program_path):
            raise ConfigurationError("You must compile the EVALB scorer before using it."
                                     " Run 'make' in the 'scripts/EVALB' directory.")

        self._header_line = ['ID', 'Len.', 'Stat.', 'Recal', 'Prec.', 'Bracket',
                             'gold', 'test', 'Bracket', 'Words', 'Tags', 'Accracy']

        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0

    @overrides
    def __call__(self, predicted_trees: List[Tree], gold_trees: List[Tree]) -> None: # type: ignore
        """
        Parameters
        ----------
        predicted_trees : ``List[Tree]``
            A list of predicted NLTK Trees to compute score for.
        gold_trees : ``List[Tree]``
            A list of gold NLTK Trees to use as a reference.
        """
        tempdir = tempfile.gettempdir()
        gold_path = os.path.join(tempdir, "gold.txt")
        predicted_path = os.path.join(tempdir, "predicted.txt")
        output_path = os.path.join(tempdir, "output.txt")
        with open(gold_path, "w") as gold_file:
            for tree in gold_trees:
                gold_file.write(f"{tree.pformat(margin=1000000)}\n")

        with open(predicted_path, "w") as predicted_file:
            for tree in predicted_trees:
                predicted_file.write(f"{tree.pformat(margin=1000000)}\n")

        command = f"{self._evalb_program_path} -p {self._evalb_param_path} " \
                  f"{gold_path} {predicted_path} > {output_path}"
        subprocess.run(command, shell=True, check=True)

        with open(output_path) as infile:
            for line in infile:
                stripped = line.strip().split()
                if len(stripped) == 12 and stripped != self._header_line:
                    # This line contains results for a single tree.
                    numeric_line = [float(x) for x in stripped]
                    self._correct_predicted_brackets += numeric_line[5]
                    self._gold_brackets += numeric_line[6]
                    self._predicted_brackets += numeric_line[7]

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average precision, recall and f1.
        """
        recall = self._correct_predicted_brackets / self._gold_brackets if self._gold_brackets > 0 else 0.0
        precision = self._correct_predicted_brackets / self._predicted_brackets if self._gold_brackets > 0 else 0.0
        f1_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        if reset:
            self.reset()
        return {"evalb_recall": recall, "evalb_precision": precision, "evalb_f1_measure": f1_measure}

    @overrides
    def reset(self):
        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0
