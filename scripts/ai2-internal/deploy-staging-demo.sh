#!/bin/bash

CONTAINER=$1

USAGE="USAGE: ./deploy-staging-demo.sh [CONTAINER]"
if [ ! -n "$CONTAINER" ] ; then
  echo "$USAGE"
  exit 1
fi

if [ "$#" -ne 1 ]; then
  echo "Too many parameters"
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
              value: "127.0.0.1"
            - name: DEMO_POSTGRES_PORT
              value: "5432"
            - name: DEMO_POSTGRES_DBNAME
              value: "demo"
            - name: DEMO_POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: cloudsql-db-credentials
                  key: username
            - name: DEMO_POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: cloudsql-db-credentials
                  key: password
          resources:
            limits:
              cpu: 1000m
              memory: 3000Mi
            # "requests" specify how much your container will be granted as a baseline.
            requests:
              cpu: 1000m
              memory: 3000Mi
          command:
            - /bin/bash
            - -c
            - "allennlp/run serve"
          readinessProbe:
            httpGet:
              path: /
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 3
        - name: cloudsql-proxy
          image: gcr.io/cloudsql-docker/gce-proxy:1.11
          command: ["/cloud_sql_proxy", "--dir=/cloudsql",
                    "-instances=ai2-general:us-central1:allennlp-demo=tcp:5432",
                    "-credential_file=/secrets/cloudsql/credentials.json"]
          volumeMounts:
            - name: cloudsql-instance-credentials
              mountPath: /secrets/cloudsql
              readOnly: true
            - name: ssl-certs
              mountPath: /etc/ssl/certs
            - name: cloudsql
              mountPath: /cloudsql
      volumes:
        - name: cloudsql-instance-credentials
          secret:
            secretName: cloudsql-instance-credentials
        - name: cloudsql
          emptyDir:
        - name: ssl-certs
          hostPath:
            path: /etc/ssl/certs
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
