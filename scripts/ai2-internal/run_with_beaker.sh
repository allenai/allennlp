#!/bin/bash

SOURCES_ARG="--source squad.latest:/squad --source glove.latest:/glove"
RESULT_ARG="--result-path /output"
GPU_ARG="--gpu-count=1"
DETACH_ARG="--detach"  # or ""

ECR_REPOSITORY=896129387501.dkr.ecr.us-west-2.amazonaws.com

PARAM_FILE=$1
EXPERIMENT_NAME=$2

COMMIT=$(git rev-parse HEAD)
IMAGE=$ECR_REPOSITORY/allennlp:$COMMIT-$RANDOM

if [ ! -n "$PARAM_FILE" ] ; then
  echo "USAGE: ./scripts/ai2-internal/run_on_beaker.sh PARAM_FILE [EXPERIMENT_NAME]"
  exit 1
fi

if [ -n "$EXPERIMENT_NAME" ] ; then
  EXPERIMENT_NAME_ARG="--name=$EXPERIMENT_NAME"
else
  EXPERIMENT_NAME_ARG=""
fi


set -e

# Get temporary ecr login. For this command to work, you need the python awscli
# package with a version more recent than 1.11.91.
eval $(aws --region=us-west-2 ecr get-login --no-include-email)

mkdir -p .beaker/
cp $PARAM_FILE .beaker/model_params.json
docker build -t $IMAGE .
docker push $IMAGE

CMD="allennlp/run train .beaker/model_params.json"

beaker experiment run $SOURCES_ARG $RESULT_ARG $EXPERIMENT_NAME_ARG $GPU_ARG $DETACH_ARG $IMAGE $CMD
