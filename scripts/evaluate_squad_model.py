import json
import os
import sys
from copy import deepcopy

import torch
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
import squad_eval

from allennlp.common import Params
from allennlp.common.params import replace_none
from allennlp.common.checks import ConfigurationError
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.models import Model
from allennlp.nn.util import arrays_to_variables, device_mapping


def main(config_file):
    config = Params.from_file(config_file)
    dataset_reader = DatasetReader.from_params(config['dataset_reader'])
    iterator_params = config['iterator']
    iterator_keys = list(iterator_params.keys())
    for key in iterator_keys:
        if key != 'batch_size':
            del iterator_params[key]
    iterator_params['type'] = 'basic'
    iterator = DataIterator.from_params(iterator_params)
    evaluation_data_path = config['validation_data_path']

    expected_version = '1.1'
    with open(evaluation_data_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        official_script_dataset = dataset_json['data']

    cuda_device = 0
    squad_eval.verbosity = 1
    model = Model.load(config, cuda_device=cuda_device)

    # Load the evaluation data
    print("Reading evaluation data from %s" % evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(model._vocab)

    model.eval()
    generator = iterator(dataset, num_epochs=1, shuffle=False)
    print("Predicting best spans for the evaluation data")
    best_spans = []
    result_dict = {}
    for batch in tqdm.tqdm(generator):
        tensor_batch = arrays_to_variables(batch, cuda_device, for_training=False)
        result = model.forward(**tensor_batch)
        best_span_tensor = result['best_span']
        for i in range(best_span_tensor.size(0)):
            best_spans.append(best_span_tensor[i].data.cpu().tolist())
    for best_span, instance in zip(best_spans, dataset.instances):
        span_tokens = instance.fields['passage'].tokens[best_span[0]:best_span[1]]
        # We have to do some hacks to get from our tokens back to the original passage text, so
        # that our answers get scored correctly.  This could be made much easier if we kept around
        # the character offset in the original text when we tokenize things.
        span_text = fix_span_text(span_tokens, instance.metadata['original_passage'])
        question_id = instance.metadata['id']
        result_dict[question_id] = span_text
    metrics = model.get_metrics()
    official_result = squad_eval.evaluate(official_script_dataset, result_dict)
    print("Our model's metrics:", metrics)
    print("Official result:", official_result)


def fix_span_text(span_tokens, passage):
    # The official evaluation ignores casing, and our model has probably lowercased the tokens.
    span_tokens = [token.lower() for token in span_tokens]
    passage = passage.lower()

    # First we'll look for inter-word punctuation and see if we should be combining tokens.
    interword_punctuation = ['-', '/', '$', '–', '€', '£', '‘', '’']
    for punct in interword_punctuation:
        keep_trying = True
        while keep_trying:
            try:
                index = span_tokens.index(punct)
                prev_num_tokens = len(span_tokens)
                span_tokens = try_combine_tokens(span_tokens, index, passage)
                keep_trying = len(span_tokens) < prev_num_tokens
            except ValueError:
                keep_trying = False
    word_initial_punctuation = ['$', '£', 'k']
    for punct in word_initial_punctuation:
        keep_trying = True
        while keep_trying:
            keep_trying = False
            for index, token in enumerate(span_tokens):
                if token.startswith(punct):
                    prev_num_tokens = len(span_tokens)
                    # Try combining this token with the previous token.
                    span_tokens = try_combine_tokens(span_tokens, index, passage, end_offset=1)
                    if len(span_tokens) < prev_num_tokens:
                        keep_trying = True
                        break
                    # Try combining this token with the next token.
                    span_tokens = try_combine_tokens(span_tokens, index, passage, start_offset=0, end_offset=2)
                    if len(span_tokens) < prev_num_tokens:
                        keep_trying = True
                        break
    word_final_punctuation = ['$']
    for punct in word_final_punctuation:
        keep_trying = True
        while keep_trying:
            keep_trying = False
            for index, token in enumerate(span_tokens):
                if token.endswith(punct):
                    prev_num_tokens = len(span_tokens)
                    # Try combining this token with the next token.
                    span_tokens = try_combine_tokens(span_tokens, index, passage, start_offset=0, end_offset=2)
                    if len(span_tokens) < prev_num_tokens:
                        keep_trying = True
                        break

    span_text = ' '.join(span_tokens)

    span_text = span_text.replace(" 's", "'s")
    span_text = span_text.replace(" ’s", "’s")
    span_text = span_text.replace("“", "")
    span_text = span_text.replace("”", "")
    span_text = ' '.join(span_text.strip().split())
    if span_text not in passage and span_text.replace('can not', 'cannot') in passage:
        span_text = span_text.replace('can not', 'cannot')
    return span_text


def try_combine_tokens(span_tokens, index, passage, start_offset = 1, end_offset = 2):
    """
    We think the three tokens centered on ``index`` should maybe be a single token.  This function
    combines them if it looks like there's reason to do so in the passage.
    """
    span_tokens = deepcopy(span_tokens)
    start_index = max(index - start_offset, 0)
    end_index = min(index + end_offset, len(span_tokens))
    tokens_to_combine = span_tokens[start_index:end_index]
    current_text = ' '.join(tokens_to_combine)
    replaced = ''.join(tokens_to_combine)
    if current_text not in passage and replaced in passage:
        span_tokens[start_index:end_index] = [replaced]
    return span_tokens


def test_fix_span_text():
    test_cases = [
            ("The kilogram-force is not a part",
             ["the", "kilogram", "-", "force", "is", "not"],
             "the kilogram-force is not"),
            ("The kilogram-force is not a part",
             ["kilogram", "-", "force"],
             "kilogram-force"),
            ("In the 1910s, New York–based filmmakers were attracted to",
             ["new", "york", "–", "based", "filmmakers"],
             "new york–based filmmakers"),
            ("offered a US$10 a week raise over Tesla's US$18 per week salary; Tesla refused",
             ["us", "$10", "a", "week", "raise", "over", "tesla", "'s", "us", "$18", "per",
              "week", "salary"],
             "us$10 a week raise over tesla's us$18 per week salary"),
            ("offered a US$10 a week raise over Tesla's US$18 per week salary; Tesla refused",
             ["us$", "10", "a", "week", "raise", "over", "tesla", "'s", "us$", "18", "per",
              "week", "salary"],
             "us$10 a week raise over tesla's us$18 per week salary"),
            ("while BSkyB paying £304m for the Premier League rights",
             ["£304", "m"],
             "£304m"),
            ("the cameras were upgraded to 5K resolution",
             ["5", "k"],
             "5k"),
            ]
    for passage, tokens, expected_text in test_cases:
        assert fix_span_text(tokens, passage) == expected_text, expected_text


if __name__ == '__main__':
    # test_fix_span_text()
    main(sys.argv[1])
