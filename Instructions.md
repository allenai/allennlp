# Getting BiDAF (Pytorch) up for experimentation on Gypsum

1\. Clone the repo.
```
git clone https://github.com/Uppaal/allennlp.git
cd allennlp
```
2\. Cut a branch by your name
```
git checkout -b <name>
```
3\. Dependencies
```
module remove python/3.5
module load python/3.6.1
```
Do not load python 3.5 back right now, because the default version is taken as 3.5. 

Then, create a virtual environment (just in case):
```
python3 -m venv <environment name>
```
To activate the environment, 
```
source <envinroment name>/bin/activate
```
To deactivate, 
```
deactivate
```
After activting the environment, run the install_requirements script and install Pytorch 0.3.1:
```
INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh
pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl 
pip3 install torchvision
```
Tesh the installation by running
```
python ./scripts/verify.py
```
4\. Datasets
```
mkdir data
```
Copy the raw versions (i.e. the directories with dev-v1.1.json  test-v1.1.json  train-v1.1.json) for SQuAD and NewsQA to the data directory. 

5\. Slurm files

Use the following script. Change the parts in <> to your running requirements.
```
#!/bin/bash
#
#SBATCH --mem=<memory requirement>
#SBATCH --job-name=1-gpu-bidaf-pytorch
#SBATCH --partition=<gpu type>
#SBATCH --output=bidaf-pytorch-%A.out
#SBATCH --error=bidaf-pytorch-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your e-mail>
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python/3.6.1
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44

## Change this line so that it points to your bidaf github folder
cd ..

# Training (Default - on SQuAD)
#python -m allennlp.run train training_config/bidaf.json -s output_path

# Evaluation (Default - on SQuAD)
python -m allennlp.run evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz --evaluation-data-file https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json

# Evaluate on NewsQA
python -m allennlp.run evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz --evaluation-data-file "data/newsqa_raw/test-v1.1.json"

# Prediction on a user defined passage
echo '{"passage": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.", "question": "How many partially reusable launch systems were developed?"}' > examples.jsonl
python -m allennlp.run predict https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz  examples.jsonl
```
6\. Options for model training and evaluation
- Change options and parameters in the training_config/bidaf.json file
- Models and results are stored in "~/output_path" Best and last model are both stored.

7\. Push your branch to master.
```
git push origin <name>
```
