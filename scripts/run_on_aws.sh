#!/bin/bash

ARISTO_BINARY=~/clone/aristo/bin/aristo
ECR_REPOSITORY=896129387501.dkr.ecr.us-west-2.amazonaws.com

CONTAINER_TAG=$1
PARAM_FILE=$2

if [ ! -n "$CONTAINER_TAG" ] || [ ! -n "$PARAM_FILE" ] ; then
  echo "USAGE: ./run_on_aws.sh [CONTAINER_TAG] [PARAM_FILE]"
  exit 1
fi


set -e

# Get temporary ecr login. For this command to work, you need the python awscli
# package with a version more recent than 1.11.91.
eval $(aws --region=us-west-2 ecr get-login --no-include-email)

docker pull $ECR_REPOSITORY/deep_qa/cuda:8

docker build -t $ECR_REPOSITORY/aristo/deep_qa:$CONTAINER_TAG . --build-arg PARAM_FILE=$PARAM_FILE
docker push $ECR_REPOSITORY/aristo/deep_qa:$CONTAINER_TAG

$ARISTO_BINARY runonce --gpu $ECR_REPOSITORY/aristo/deep_qa:$CONTAINER_TAG
