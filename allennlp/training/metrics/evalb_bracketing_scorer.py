
from typing import List
import os
import tempfile
import re
import math
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

        self._recall_regex = re.compile(r"Bracketing Recall\s+=\s+(\d+\.\d+)")
        self._precision_regex = re.compile(r"Bracketing Precision\s+=\s+(\d+\.\d+)")
        self._f1_measure_regex = re.compile(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)")

        self._precision = 0.0
        self._recall = 0.0
        self._f1_measure = 0.0
        self._count = 0.0

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

        recall = math.nan
        precision = math.nan
        fmeasure = math.nan
        with open(output_path) as infile:
            for line in infile:
                recall_match = self._recall_regex.match(line)
                if recall_match:
                    recall = float(recall_match.group(1))
                precision_match = self._precision_regex.match(line)
                if precision_match:
                    precision = float(precision_match.group(1))
                f1_measure_match = self._f1_measure_regex.match(line)
                if f1_measure_match:
                    fmeasure = float(f1_measure_match.group(1))
                    break
        if any([math.isnan(recall), math.isnan(precision)]):
            raise RuntimeError(f"Call to EVALB produced invalid metrics: recall: "
                               f"{recall}, precision: {precision}, fmeasure: {fmeasure}")

        if math.isnan(fmeasure) and recall == 0.0 and precision == 0.0:
            fmeasure = 0.0
        elif math.isnan(fmeasure):
            raise RuntimeError(f"Call to EVALB produced an invalid f1 measure, "
                               f"which was not due to zero division: recall: "
                               f"{recall}, precision: {precision}, fmeasure: {fmeasure}")

        self._precision += precision / 100.0
        self._recall += recall / 100.0
        self._f1_measure += fmeasure / 100.0
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average precision, recall and f1.
        """
        metrics = {}
        metrics["evalb_precision"] = self._precision / self._count if self._count > 0 else 0.0
        metrics["evalb_recall"] = self._recall / self._count if self._count > 0 else 0.0
        metrics["evalb_f1_measure"] = self._f1_measure / self._count if self._count > 0 else 0.0

        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self):
        self._recall = 0.0
        self._precision = 0.0
        self._f1_measure = 0.0
        self._count = 0
