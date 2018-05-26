# AllenNLP Conda Builds

This document is intended to give you a primer on the AllenNLP conda build system.

## Background: Conda Recipes

Conda packages are built with a tool called `conda-build`, which takes in a recipe. Packaging
code for conda mostly entails writing the recipe (a file called `meta.yaml`). 
`conda-forge`, a community organization that serves to aggregate and maintain 
community-authored recipes, has [a great tutorial on how to write 
recipes](https://conda-forge.org/docs/recipe.html). I'd also recommend taking a look at their
[example recipe](https://github.com/conda-forge/staged-recipes/blob/master/recipes/example/meta.yaml).

Let's go through the AllenNLP recipe (as of May 2018) and take a look at its components:

```
# Recipes support Jinja templating, so this is basically a variable that we can use later. 
# This makes maintenance easier, since usually all you have to do is increment the version
# and modify the sha256 hash of the tarball for the new release.
{% set name = "allennlp" %}
{% set version = "0.4.3" %}
{% set file_ext = "tar.gz" %}
{% set hash_type = "sha256" %}
{% set hash_value = "dbc0b7de268a14ad978ad29ceb785c0a22cab13e2786f04fe933b4c71e9ad7aa" %}
# sha256 is the prefered checksum -- you can get it for a file with:
#  `openssl sha256 <file name>`.

# Defines the name of the package
package:
  name: '{{ name|lower }}'
  version: '{{ version }}'

# This section defines where we get the package from and how to validate it (hash).
source:
  url: https://github.com/allenai/allennlp/archive/v{{ version }}.tar.gz
  '{{ hash_type }}': '{{ hash_value }}'

# This section defines how the build is carried out.
build:
  # We can optionally skip the build for certain platforms with comments.
  # Here, we're skipping the build if the platform is Windows (win) or if
  # the Python version is pre-3.6 (py<36).
  # A full list of variables you can use is here: 
  # https://conda.io/docs/user-guide/tasks/build-packages/define-metadata.html#preprocess-selectors
  skip: True  # [win or py<36]
  # The build number. This helps ensure that each build (even within versions) is unique.
  # If you update the recipe without incrementing the version, you should increment the number.
  # When upgrading the version in the recipe, you should set the number back to 0.
  number: 0
  # If the installation is complex, or different between Unix and Windows, use separate bld.bat and build.sh files instead of this key.
  # This should work for pure-python packages, though.
  script: python -m pip install --no-deps --ignore-installed .

requirements:
  # Requirements necessary to build. We need python and pip, because they're used in the "script" step above.
  build:
    - pip
    - python
  # Requirements necessary to _run_. Note that we don't need something like pytorch to simply package the code,
  # but if we were to run the code, we'd need to use it.
  run:
    - python
    - pytorch ==0.3.1
    - pyhocon ==0.3.35
    - typing
    - overrides
    - nltk
    - spacy >=2.0,<2.1
    - numpy
    - tensorboardx ==1.0
    - cffi ==1.11.2
    - awscli >=1.11.91
    - flask ==0.12.1
    - flask-cors ==3.0.3
    - gevent ==1.2.2
    - psycopg2
    - requests >=2.18
    - tqdm >=4.19
    - editdistance
    - h5py
    - scikit-learn
    - scipy
    - pytz ==2017.3
    - unidecode
    - wget
    - perl
    - openjdk

# This section defines tests that are run after the package is built.
test:
  # Try importing these modules.
  imports:
    - allennlp
    - allennlp.commands
    - allennlp.common
    - allennlp.common.testing
    - allennlp.data
    - allennlp.data.dataset_readers
    - allennlp.data.dataset_readers.coreference_resolution
    - allennlp.data.dataset_readers.dataset_utils
    - allennlp.data.dataset_readers.reading_comprehension
    - allennlp.data.fields
    - allennlp.data.iterators
    - allennlp.data.token_indexers
    - allennlp.data.tokenizers
    - allennlp.models
    - allennlp.models.coreference_resolution
    - allennlp.models.encoder_decoders
    - allennlp.models.reading_comprehension
    - allennlp.models.semantic_parsing
    - allennlp.models.semantic_parsing.nlvr
    - allennlp.models.semantic_parsing.wikitables
    - allennlp.modules
    - allennlp.modules.seq2seq_encoders
    - allennlp.modules.seq2vec_encoders
    - allennlp.modules.similarity_functions
    - allennlp.modules.span_extractors
    - allennlp.modules.text_field_embedders
    - allennlp.modules.token_embedders
    - allennlp.nn
    - allennlp.nn.decoding
    - allennlp.nn.decoding.decoder_trainers
    - allennlp.nn.regularizers
    - allennlp.semparse
    - allennlp.semparse.contexts
    - allennlp.semparse.type_declarations
    - allennlp.semparse.worlds
    - allennlp.service
    - allennlp.service.predictors
    - allennlp.training
    - allennlp.training.metrics
  # The packages that the tests require. Note that these are here because they aren't needed
  # to run the code as well.
  requires:
    - pytest
    - flaky
    - responses >=0.7
    - jupyter
  # The files we want to add to the working directory when running tests.
  source_files:
    - tests
    - tutorials
    - scripts
  # Test commands, run in sequence. Here, we download NLTK and SpaCy models, then run pytest.
  commands:
    - 'python -m nltk.downloader punkt'
    - 'curl -L https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz -o ~/en_core_web_sm-2.0.0.tar.gz'
    - 'tar -xf ~/en_core_web_sm-2.0.0.tar.gz -C ~'
    - 'python -m spacy link ~/en_core_web_sm-2.0.0/en_core_web_sm en_core_web_sm'
    - 'py.test -v tests/'
    - 'rm ~/en_core_web_sm-2.0.0.tar.gz'
    - 'rm -rf ~/en_core_web_sm-2.0.0'

# Package metadata
about:
  home: https://github.com/allenai/allennlp
  license: Apache 2.0
  license_family: Apache
  license_file: LICENSE
  summary: An open-source NLP research library, built on PyTorch.
  description: An open-source NLP research library, built on PyTorch.
  doc_url: 'https://allenai.github.io/allennlp-docs/'
  dev_url: 'https://github.com/allenai/allennlp'

extra:
  recipe-maintainers:
    - nelson-liu
```

For a full reference on what can go in `meta.yaml`, see 
[the conda docs](https://conda.io/docs/user-guide/tasks/build-packages/define-metadata.html#).

## Background: How do you generate the package from the recipe?

To generate the package from the recipe, you use the `conda build` tool. However, manually building 
packages is arduous since we need to repeat this process for each platform we're targeting. Luckily,
conda-forge has built some pretty great infastructure that makes it easy to set up a repository
containing a recipe and all related files needed to automatically render the binaries on public CI.
This repo is referred to as a "feedstock". Whenever pushes are made to the feedstock, CI will automatically
kick in and build the recipe from scratch and automatically upload it to conda.

[`conda-smithy`](https://github.com/conda-forge/conda-smithy) takes a recipe and generates a feedstock
that's ready for use.

To make a new feedstock, follow their instructions here: https://github.com/conda-forge/conda-smithy#making-a-new-feedstock

Periodically, you might need to update the feedstock itself (and not just the recipe, which you can directly change) --- 
for example, you might want to refresh a private token you have or change the platforms the package is built on. This
process is called "rerendering", and you can read about how to do it here: https://github.com/conda-forge/conda-smithy#re-rendering-an-existing-feedstock

## Configuring the feedstock for AllenNLP.

`conda-smithy` takes in a configuration file (`conda-forge.yml`) that affects how some components are rendered.
Here's what the AllenNLP configuration looks like:

```
channels:
  # The target [channel, label] to upload to. main is usually ok for a label.
  targets:
    - [allennlp, main]
  # The channels to pull packages from when building allennlp. By default, 
  # the included channels are 'conda-forge' and 'defaults'. We add the pytorch
  # channel since windows / osx pytorch isn't on 'conda-forge' or 'defaults' yet.
  sources:
  - pytorch
  - conda-forge
  - defaults
travis:
  secure:
    BINSTAR_TOKEN: the token used to interact with travis
appveyor:
  secure:
    BINSTAR_TOKEN: the token used to interact with appveyor
# The image used in CI to build the package.
docker:
  image: allennlp/allennlp-conda-linux-anvil
```

## Why do we need to use a custom image?

By default, the Docker image used by CI is [`condaforge/linux-anvil`](https://hub.docker.com/r/condaforge/linux-anvil/).
Unfortunately, we can't use this as is because we pull PyTorch for OSX (and eventually Windows) from the `pytorch` channel.
These official pytorch binaries only support [`glibc 2.17`](https://github.com/pytorch/pytorch/issues/6607#issuecomment-381707146), 
which centOS 6 (the base image used in `condaforge/linux-anvil`) does not have (it has `glibc 2.12`).

Apparently, updating glibc is not something that people do and it's purported to be pretty arduous. The easiest 
solution was thus to just edit the `condaforge/linux-anvil` image to use CentOS7. The result of this edit lives
in https://github.com/allenai/allennlp-conda-linux-anvil , or `allennlp/allennlp-conda-linux-anvil` on Docker hub.
Thus, to actually use this image at build time, we need to add it to the `conda-forge.yml` configuration.

## Making a new release

This is as simple as incrementing the version counter, editing the expected SHA256 hash, and then updating the requirements
or tests as necessary (e.g. testing that importing a new module works). When you push, CI won't upload the images _unless_
all the tests pass, which prevents accidental mispackaging.
