# allennlp.training.metrics.evalb_bracketing_scorer

## EvalbBracketingScorer
```python
EvalbBracketingScorer(self, evalb_directory_path:str='/Users/markn/allen_ai/allennlp/allennlp/tools/EVALB', evalb_param_filename:str='COLLINS.prm', evalb_num_errors_to_kill:int=10) -> None
```

This class uses the external EVALB software for computing a broad range of metrics
on parse trees. Here, we use it to compute the Precision, Recall and F1 metrics.
You can download the source for EVALB from here: <https://nlp.cs.nyu.edu/evalb/>.

Note that this software is 20 years old. In order to compile it on modern hardware,
you may need to remove an ``include <malloc.h>`` statement in ``evalb.c`` before it
will compile.

AllenNLP contains the EVALB software, but you will need to compile it yourself
before using it because the binary it generates is system dependent. To build it,
run ``make`` inside the ``allennlp/tools/EVALB`` directory.

Note that this metric reads and writes from disk quite a bit. You probably don't
want to include it in your training loop; instead, you should calculate this on
a validation set only.

Parameters
----------
evalb_directory_path : ``str``, required.
    The directory containing the EVALB executable.
evalb_param_filename : ``str``, optional (default = "COLLINS.prm")
    The relative name of the EVALB configuration file used when scoring the trees.
    By default, this uses the COLLINS.prm configuration file which comes with EVALB.
    This configuration ignores POS tags and some punctuation labels.
evalb_num_errors_to_kill : ``int``, optional (default = "10")
    The number of errors to tolerate from EVALB before terminating evaluation.

### get_metric
```python
EvalbBracketingScorer.get_metric(self, reset:bool=False)
```

Returns
-------
The average precision, recall and f1.

### reset
```python
EvalbBracketingScorer.reset(self)
```

Reset any accumulators or internal state.

