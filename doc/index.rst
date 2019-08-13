.. AllenNLP documentation master file, created by
   sphinx-quickstart on Mon Aug  7 09:11:08 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

Built on PyTorch, AllenNLP makes it easy to design and evaluate new deep
learning models for nearly any NLP problem, along with the infrastructure to
easily run them in the cloud or on your laptop.  AllenNLP was designed with the
following principles:

* *Hyper-modular and lightweight.* Use the parts which you like seamlessly with PyTorch.
* *Extensively tested and easy to extend.* Test coverage is above 90% and the example
  models provide a template for contributions.
* *Take padding and masking seriously*, making it easy to implement correct
  models without the pain.
* *Experiment friendly.*  Run reproducible experiments from a json
  specification with comprehensive logging.

AllenNLP includes reference implementations of high quality models for Semantic
Role Labelling, Question and Answering (BiDAF), Entailment (decomposable
attention), and more (see https://allennlp.org/models).

AllenNLP is built and maintained by the Allen Institute for Artificial
Intelligence, in close collaboration with researchers at the University of
Washington and elsewhere. With a dedicated team of best-in-field researchers
and software engineers, the AllenNLP project is uniquely positioned to provide
state of the art models with high quality engineering.

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   api/allennlp.commands
   api/allennlp.common
   api/allennlp.data
   api/allennlp.interpret
   api/allennlp.models
   api/allennlp.predictors
   api/allennlp.modules
   api/allennlp.nn
   api/allennlp.semparse
   api/allennlp.service
   api/allennlp.state_machines
   api/allennlp.tools
   api/allennlp.training
   api/allennlp.pretrained




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
