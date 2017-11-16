#!/bin/bash

CONTAINER=$1

USAGE="USAGE: ./deploy-staging-demo.sh [CONTAINER]"
if [ ! -n "$CONTAINER" ] ; then
  echo "$USAGE"
  exit 1
fi

echo "Deploying container '$CONTAINER' to staging."


kubectl apply -f - <<EOF
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  name: allennlp-demo-staging
  namespace: allennlp
  labels:
    contact: allennlp
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: allennlp-demo-staging
    spec:
      containers:
        - name: allennlp
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
  name: allennlp-demo-staging
  namespace: allennlp
spec:
  type: LoadBalancer
  selector:
    app: allennlp-demo-staging
  ports:
    - port: 80
      targetPort: 8000
EOF
