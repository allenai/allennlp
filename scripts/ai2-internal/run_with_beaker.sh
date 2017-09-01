#!/bin/bash

SOURCES_ARG="--source squad.latest:/squad --source glove.latest:/glove"
RESULT_ARG="--result-path /output"
GPU_ARG="--gpu-count=1"
DETACH_ARG="--detach"  # or ""

ECR_REPOSITORY=896129387501.dkr.ecr.us-west-2.amazonaws.com

PARAM_FILE=$1
EXPERIMENT_NAME=$2

# TODO(matt): if beaker makes it possible to run experiments from the web UI, we probably should
# just have a standard image, that we don't have to rebuild each time, and we can remove this
# $RANDOM stuff.
COMMIT=$(git rev-parse HEAD)
IMAGE=$ECR_REPOSITORY/allennlp/allennlp:$COMMIT-$RANDOM

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

docker build -t $IMAGE .
docker push $IMAGE

CONFIG_DATASET_ID=$(beaker dataset create --quiet $PARAM_FILE)
FILENAME=$(basename $PARAM_FILE)
SOURCES_ARG="$SOURCES_ARG --source $CONFIG_DATASET_ID:/config"

CMD="allennlp/run train /config/$FILENAME -s /output"

beaker experiment run $SOURCES_ARG $RESULT_ARG $EXPERIMENT_NAME_ARG $GPU_ARG $DETACH_ARG $IMAGE -- $CMD
