SRC = allennlp

MD_DOCS_ROOT = docs/
MD_DOCS_API_ROOT = $(MD_DOCS_ROOT)api/
MD_DOCS_SRC = $(filter-out $(SRC)/__main__.py %/__init__.py $(SRC)/version.py,$(shell find $(SRC) -type f -name '*.py' | grep -v -E 'tests/'))
MD_DOCS = $(subst .py,.md,$(subst $(SRC)/,$(MD_DOCS_API_ROOT),$(MD_DOCS_SRC)))
MD_DOCS_CMD = python scripts/py2md.py
MD_DOCS_CONF = mkdocs.yml
MD_DOCS_CONF_SRC = mkdocs-skeleton.yml
MD_DOCS_TGT = site/
MD_DOCS_EXTRAS = $(addprefix $(MD_DOCS_ROOT),README.md CHANGELOG.md CONTRIBUTING.md)

DOCKER_TAG = latest
DOCKER_IMAGE_NAME = allennlp/allennlp:$(DOCKER_TAG)
DOCKER_TEST_IMAGE_NAME = allennlp/test:$(DOCKER_TAG)
DOCKER_RUN_CMD = docker run --rm \
		-v $$HOME/.allennlp:/root/.allennlp \
		-v $$HOME/.cache/torch:/root/.cache/torch \
		-v $$HOME/nltk_data:/root/nltk_data

ifeq ($(shell uname),Darwin)
ifeq ($(shell which gsed),)
$(error Please install GNU sed with 'brew install gnu-sed')
else
SED = gsed
endif
else
SED = sed
endif

.PHONY : version
version :
	@python -c 'from allennlp.version import VERSION; print(f"AllenNLP v{VERSION}")'

.PHONY : check-for-cuda
check-for-cuda :
	@python -c 'import torch; assert torch.cuda.is_available(); print("Cuda is available")'

#
# Testing helpers.
#

.PHONY : lint
lint :
	flake8 .

.PHONY : format
format :
	black --check .

.PHONY : typecheck
typecheck :
	mypy . \
		--ignore-missing-imports \
		--no-strict-optional \
		--no-site-packages \
		--cache-dir=/dev/null

.PHONY : test
test :
	pytest --color=yes -rf --durations=40

.PHONY : test-with-cov
test-with-cov :
	pytest --color=yes -rf --durations=40 \
			--cov-config=.coveragerc \
			--cov=$(SRC) \
			--cov-report=xml

.PHONY : gpu-test
gpu-test : check-for-cuda
	pytest --color=yes -v -rf -m gpu

.PHONY : benchmarks
benchmarks :
	pytest -c benchmarks/pytest.ini benchmarks/

#
# Setup helpers
#

.PHONY : install
install :
	# Making sure the typing backport isn't installed.
	pip uninstall -y typing
	# Ensure pip, setuptools, and wheel are up-to-date.
	pip install --upgrade pip setuptools wheel
	# Due to a weird thing with pip, we may need egg-info before running `pip install -e`.
	# See https://github.com/pypa/pip/issues/4537.
	python setup.py install_egg_info
	# Install allennlp as editable and all dependencies.
	pip install --upgrade --upgrade-strategy eager -e . -r dev-requirements.txt
	# The above command might install the typing backport because of pydoc-markdown,
	# so we have to uninstall it again.
	pip uninstall -y typing

#
# Documention helpers.
#

.PHONY : build-all-api-docs
build-all-api-docs :
	@$(MD_DOCS_CMD) $(subst /,.,$(subst .py,,$(MD_DOCS_SRC))) -o $(MD_DOCS)

.PHONY : build-docs
build-docs : build-all-api-docs $(MD_DOCS_CONF) $(MD_DOCS) $(MD_DOCS_EXTRAS)
	mkdocs build

.PHONY : serve-docs
serve-docs : build-all-api-docs $(MD_DOCS_CONF) $(MD_DOCS) $(MD_DOCS_EXTRAS)
	mkdocs serve --dirtyreload

.PHONY : update-docs
update-docs : $(MD_DOCS) $(MD_DOCS_EXTRAS)

$(MD_DOCS_ROOT)README.md : README.md
	cp $< $@
	# Alter the relative path of the README image for the docs.
	$(SED) -i '1s/docs/./' $@

$(MD_DOCS_ROOT)%.md : %.md
	cp $< $@

$(MD_DOCS_CONF) : $(MD_DOCS_CONF_SRC) $(MD_DOCS)
	python scripts/build_docs_config.py $@ $(MD_DOCS_CONF_SRC) $(MD_DOCS_ROOT) $(MD_DOCS_API_ROOT)

$(MD_DOCS_API_ROOT)%.md : $(SRC)/%.py scripts/py2md.py
	mkdir -p $(shell dirname $@)
	$(MD_DOCS_CMD) $(subst /,.,$(subst .py,,$<)) --out $@

.PHONY : clean
clean :
	rm -rf $(MD_DOCS_TGT)
	rm -rf $(MD_DOCS_API_ROOT)
	rm -f $(MD_DOCS_ROOT)*.md
	rm -rf .pytest_cache/
	rm -rf allennlp.egg-info/
	rm -rf dist/
	rm -rf build/
	find . | grep -E '(\.mypy_cache|__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

#
# Docker helpers.
#

.PHONY : docker-image
docker-image :
	docker build \
			--pull \
			-f Dockerfile \
			-t $(DOCKER_IMAGE_NAME) .

.PHONY : docker-run
docker-run :
	$(DOCKER_RUN_CMD) $(DOCKER_IMAGE_NAME) $(ARGS)

.PHONY : docker-test-image
docker-test-image :
	docker build --pull -f Dockerfile.test -t $(DOCKER_TEST_IMAGE_NAME) .

.PHONY : docker-test-run
docker-test-run :
	$(DOCKER_RUN_CMD) --gpus 2 $(DOCKER_TEST_IMAGE_NAME) $(ARGS)
