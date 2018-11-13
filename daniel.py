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

def create_instance(reader, question, paragraph, answer_start, answer_end):
    return reader.text_to_instance(question, paragraph, zip([answer_start], [answer_end]))

def solve_sample_question():
    model, dataset_reader = load_model()

    instance1 = create_instance(reader=dataset_reader, question = "What kind of test succeeded on its first attempt?",
                    paragraph="One time I was writing a unit test, and it succeeded on the first attempt.",
                                answer_start=1, answer_end=5)

    instance2 = create_instance(reader=dataset_reader, question="What happens when you go cheap?",
                                paragraph="Dion Lewis on Blowout Win vs. Patriots: 'That's What Happens When You Go Cheap'.",
                                answer_start=5, answer_end=6)

    instances = [instance1] # instance2
    dataset = Batch(instances)
    dataset.index_instances(model.vocab)
    cuda_device = model._get_prediction_device()
    model_input = dataset.as_tensor_dict(verbose=True)
    outputs = model(**model_input)
    print(outputs["best_span_str"])

if __name__ == "__main__":
    solve_sample_question()

