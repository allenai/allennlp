config = configs/bidaf.yml
name = $(shell basename)

ALLENNLP_REPO = allenai/allennlp
ALLENNLP_REV = $(shell git ls-remote https://github.com/$(ALLENNLP_REPO).git | head -1 | sed 's/[[:space:]]*HEAD//')
ALLENNLP = git+https://github.com/$(ALLENNLP_REPO).git@$(ALLENNLP_REV)

MODELS_REPO = allenai/allennlp-models
MODELS_REV = $(shell git ls-remote https://github.com/$(MODELS_REPO).git | head -1 | sed 's/[[:space:]]*HEAD//')
MODELS = git+https://github.com/$(MODELS_REPO).git@$(MODELS_REV)

DATE_ID := $(shell date -u +%Y-%m-%d-%H-%M-%S)
IMAGE_NAME := allennlp-with-models-$(DATE_ID)

ifeq ($(shell uname -s),Darwin)
	SED_CMD = sed -i ''
else
	SED_CMD = sed -i
endif

.PHONY : image
image :
	@echo "Building docker image..."
	@docker build \
		--pull \
		--build-arg ALLENNLP=$(ALLENNLP) \
		--build-arg MODELS=$(MODELS) \
		-f Dockerfile \
		-t allennlp-with-models .
	@echo "Testing docker image..."
	@docker run --rm allennlp-with-models test-install

.SECONDARY :
/tmp/beaker/configs/%.yml : image
	@echo "Creating beaker config with:"
	@echo "    image :           $(IMAGE_NAME)"
	@echo "    allennlp :        $(ALLENNLP)"
	@echo "    allennlp-models : $(ALLENNLP)"
	@mkdir -p /tmp/beaker/configs
	@cp configs/$*.yml $@
	@$(SED_CMD) 's/IMAGE_NAME/$(IMAGE_NAME)/g' $@
	@$(SED_CMD) 's|ALLENNLP_VERSION|$(ALLENNLP)|g' $@
	@$(SED_CMD) 's|ALLENNLP_MODELS_VERSION|$(MODELS)|g' $@
	@echo "Beaker config created: $@"

configs/%.yml : /tmp/beaker/configs/%.yml
	@echo "Uploading image to beaker as $(IMAGE_NAME)"
	# @beaker image create -n $(IMAGE_NAME) allennlp-with-models
	@echo "Submitting experiment to beaker as $*_$(subst -,_,$(DATE_ID))"
	# @beaker experiment create --name $*_$(subst -,_,$(DATE_ID)) -f $< --workspace ai2/AllenNLPPerf

FORCE :
