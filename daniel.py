import json
import logging
import os
import sys

import numpy
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from scipy import linalg
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch
from sklearn import cluster, metrics
import time as time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from scipy.cluster.hierarchy import dendrogram, linkage

def load_model():
    from allennlp.models import load_archive
    archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
    # archive = load_archive("finetune_factor_001_epoch_2/model.tar.gz")
    config = archive.config.duplicate()
    model = archive.model
    model.eval()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    return model, dataset_reader


def solve_sample_question():
    model, dataset_reader = load_model()

    question = "What kind of test succeeded on its first attempt?"
    paragraph = "One time I was writing a unit test, and it succeeded on the first attempt."

    a = solve(question, paragraph, model, dataset_reader, ["unit test"])
    print("")

def solve(question, paragraph, model, dataset_reader, answers):
    print(question)
    print(paragraph)
    print(answers)
    instance = dataset_reader.text_to_instance(question, paragraph)
    instances = [instance]
    dataset = Batch(instances)
    dataset.index_instances(model.vocab)
    cuda_device = model._get_prediction_device()
    model_input = dataset.as_tensor_dict()
    outputs = model(**model_input)
    # with open('ipython/squad_dev_with_prereq/out22-ner-test.txt', 'a') as ff:
    #     ff.write(question.replace('\n', ' ') + "\n" + paragraph.replace('\n', ' ') + "\n" + str(json.dumps(answers)) + "\n")

if __name__ == "__main__":
    solve_sample_question()

