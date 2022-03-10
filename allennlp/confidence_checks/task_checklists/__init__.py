import warnings

try:
    from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
    from allennlp.confidence_checks.task_checklists.sentiment_analysis_suite import (
        SentimentAnalysisSuite,
    )
    from allennlp.confidence_checks.task_checklists.question_answering_suite import (
        QuestionAnsweringSuite,
    )
    from allennlp.confidence_checks.task_checklists.textual_entailment_suite import (
        TextualEntailmentSuite,
    )
except ImportError:
    warnings.warn(
        "The checklist integration of allennlp is optional; if you're using conda, "
        "it can be installed with `conda install allennlp-checklist`, "
        "otherwise use `pip install allennlp[checklist]`."
    )
    raise
