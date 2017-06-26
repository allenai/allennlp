[![Build Status](https://travis-ci.org/allenai/allennlp.svg?branch=master)](https://travis-ci.org/allenai/allennlp)

# AllenNLP

A [Apache 2.0](https://github.com/allenai/allennlp/blob/master/LICENSE) natural language processing toolkit using state-of-the-art deep learning models.


### Setting up a development environment

DeepQA is built using Python 3.  The easiest way to set up a compatible
environment is to use [Conda](https://conda.io/).  This will set up a virtual
environment with the exact version of Python used for development along with all the
dependencies needed to run DeepQA.

1.  [Download and install Conda](https://conda.io/docs/download.html).
2.  Create a Conda environment with Python 3.

    ```
    conda create -n allennlp python=3.5
    ```

3.  Now activate the Conda environment.

    ```
    source activate allennlp
    ```

4.  Install the required dependencies.

    ```
    ./scripts/install_requirements.sh
    ```

5. Visit http://pytorch.org/ and install the relevant pytorch package.

6.  Set the `PYTHONHASHSEED` for repeatable experiments.

    ```
    export PYTHONHASHSEED=2157
    ```

You should now be able to test your installation with `pytest -v`.  Congratulations!

### Setting up a Kubernetes development environment

1. Follow the instructions for installing and setting up
[Kubernetes](https://github.com/allenai/infrastructure/tree/master/kubernetes).

2. Fill in the [yaml file](./kubernetes-dev-machine.yaml). You need to add:

    - The name of the job, under the `metadata:` heading.
    - The namespace you wish to run in under the `metadata:` heading. To see which namespaces
      are available, run ` kubectl get ns ` .
    - Your contact name (first bit of your email) under `labels.contact:`.

3. Run `kubectl create -f /path/to/kubernetes-dev-machine.yaml`. This creates your job on the cluster.

4. Retrieve the name of the pod created to run your job using `kubectl get pods --namespace <NAMESPACE>`.
   This will be the name you provided for your job above, plus some random characters.

5. Get a shell inside the container using `kubectl exec -it <PODNAME> --container dev-environment -- /bin/bash`

6. When you are done, don't forget to kill your job using `kubectl delete -f /path/to/kubernetes-dev-machine.yaml`


