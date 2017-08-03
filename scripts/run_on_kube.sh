#!/bin/python

ECR_REPOSITORY=896129387501.dkr.ecr.us-west-2.amazonaws.com

PARAM_FILE=$1
CONTACT=$2
CONTAINER_TAG=$(git rev-parse HEAD)
IMAGE=$ECR_REPOSITORY/allennlp/allennlp-gpu:$CONTAINER_TAG
ID=$(openssl rand -base64 6)

USAGE="USAGE: ./run_on_kube.sh [PARAM_FILE] [CONTACT]"
if [ ! -n "$PARAM_FILE" ] ; then
  echo "$USAGE"
  exit 1
fi

if [ ! -n "$CONTACT" ] ; then
  echo "$USAGE"
  exit 1
fi

echo "Configuration:"
echo "  PARAM_FILE: $PARAM_FILE"
echo "  Contact:    $CONTACT"
echo "  Image:      $IMAGE"

git diff-index --quiet HEAD --
ec=$?
if [ $ec -eq 1 ] ; then
  echo "Your git repository has outstanding changes."
  echo "Please commit your changes before running an experiment."
  exit 1
fi

set -e

# Get temporary ecr login. For this command to work, you need the python awscli
# package with a version more recent than 1.11.91.
echo "Logging in to AWS ECR..."
eval $(aws --region=us-west-2 ecr get-login --no-include-email)

echo "Building the Docker image..."
docker build -f Dockerfile.gpu -t $IMAGE . --build-arg PARAM_FILE=$PARAM_FILE
echo "Pushing the Docker image to ECR..."
docker push $IMAGE

COMMAND="bash -c touch /net/efs/aristo/helloworld"

cat >spec.yaml <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: allennlp-dev-env-$ID
  namespace: allennlp
  labels:
    contact: $CONTACT
spec:
  template:
    metadata:
      labels:
        autoscaler-task: "yes"
    spec:
      restartPolicy: Never
      containers:
        - name: allennlp-dev-env
          image: $IMAGE
          stdin: true
          tty: true
          resources:
            requests:
              # p2.xlarge has ~55GB free memory, ~4 cores of CPU. We request most of that.
              cpu: 3000m
              memory: 50Gi
            # "limits" specify the max your container will be allowed to use.
            limits:
              # Set this to the number of GPUs you want to use. Note that if you set this higher
              # than the maxiumum available on our GPU instances, your job will never get scheduled.
              # Note that this can ONLY appear in "limits" - "requests" should not have GPU
              # resources.
              # See https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/ for more info.
              alpha.kubernetes.io/nvidia-gpu: 1
              # Set the limit slightly higher than the request for CPU.
              cpu: 3500m
              memory: 50Gi
          volumeMounts:
          - name: nvidia-driver
            mountPath: /usr/local/nvidia
            readOnly: true
          - name: nfs
            mountPath: /net/efs/aristo
            readOnly: true
          command:
             - $COMMAND
      volumes:
      # Mount in the GPU driver from the host machine. This guarantees compatibility.
      - name: nvidia-driver
        hostPath:
          path: /var/lib/nvidia-docker/volumes/nvidia_driver/367.57
      # Mount the efs drive
      - name: nfs
        hostPath:
            path: /net/efs/aristo
      tolerations:
        # This is required to run on our GPU machines. Do not remove.
        - key: "GpuOnly"
          operator: "Equal"
          value: "true"
EOF

echo "Sending job to Kubernetes..."
kubectl create -f spec.yaml

echo "Done!"
