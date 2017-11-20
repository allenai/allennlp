#!/bin/bash

CONTAINER=$1
FORCE=$2

USAGE="USAGE: ./deploy-production-demo.sh [CONTAINER] [--force]"
if [ ! -n "$CONTAINER" ] ; then
  echo "$USAGE"
  exit 1
fi

if [ "$#" -gt 2 ]; then
  echo "Too many parameters"
  echo "$USAGE"
  exit 1
fi

DRYRUN="--dry-run"
if [ ! -z $FORCE ] ; then
  if [ $FORCE = "--force" ] ; then
    DRYRUN=""
    echo "Deploying container '$CONTAINER' to production."
  else
    echo "$USAGE"
    exit 1
  fi
else
  echo "Deploying container '$CONTAINER' to production. (dry run)"
fi

kubectl apply $DRYRUN -f - <<EOF
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  name: allennlp-demo-prod
  namespace: allennlp
  labels:
    contact: allennlp
spec:
  replicas: 4
  template:
    metadata:
      labels:
        app: allennlp-demo-prod
    spec:
      containers:
        - name: allennlp-demo-prod
          image: "$CONTAINER"
          # See
          # https://kubernetes.io/docs/concepts/configuration/manage-compute-resources-container/
          # for documentation on the resources section.
          env:
            - name: DEMO_POSTGRES_HOST
              valueFrom:
                secretKeyRef:
                  name: allennlp-demo-postgres
                  key: host
            - name: DEMO_POSTGRES_DBNAME
              valueFrom:
                secretKeyRef:
                  name: allennlp-demo-postgres
                  key: dbname
            - name: DEMO_POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: allennlp-demo-postgres
                  key: user
            - name: DEMO_POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: allennlp-demo-postgres
                  key: password
          resources:
            limits:
              cpu: 1000m
              memory: 2000Mi
            # "requests" specify how much your container will be granted as a baseline.
            requests:
              cpu: 1000m
              memory: 2000Mi
          command:
            - /bin/bash
            - -c
            - "allennlp/run serve"
---
apiVersion: v1
kind: Service
metadata:
  name: allennlp-demo-prod
  namespace: allennlp
spec:
  type: LoadBalancer
  selector:
    app: allennlp-demo-prod
  ports:
    - port: 80
      targetPort: 8000
EOF
