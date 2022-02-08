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
        'To use this integration you should install ``allennlp`` with the "checklist" extra '
        "(e.g. ``pip install allennlp[checklist]``) or just install checklist after the fact."
    )
    raise
