import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.DEBUG)
from allennlp.commands.train import datasets_from_params
from allennlp.common import Params
from allennlp.data import Instance


def main(params: Params, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    params['dataset_reader']['include_table_metadata'] = True
    if 'validation_dataset_reader' in params:
        params['validation_dataset_reader']['include_table_metadata'] = True
    all_datasets = datasets_from_params(params)
    for name, dataset in all_datasets.items():
        with open(outdir + name + '.jsonl', 'w') as outfile:
            for instance in iter(dataset):
                outfile.write(to_json_line(instance) + '\n')


def to_json_line(instance: Instance):
    json_obj = {}
    question_tokens = instance.fields['question'].tokens
    json_obj['question_tokens'] = [{'text': token.text, 'lemma': token.lemma_}
                                   for token in question_tokens]
    json_obj['table_lines'] = instance.fields['table_metadata'].metadata

    action_map = {i: action.rule for i, action in enumerate(instance.fields['actions'].field_list)}

    if 'target_action_sequences' in instance.fields:
        targets = []
        for target_sequence in instance.fields['target_action_sequences'].field_list:
            targets.append([])
            for target_index_field in target_sequence.field_list:
                targets[-1].append(action_map[target_index_field.sequence_index])

        json_obj['target_action_sequences'] = targets

    json_obj['example_lisp_string'] = instance.fields['example_lisp_string'].metadata

    entity_texts = []
    for entity_text in instance.fields['table'].entity_texts:
        tokens = [{'text': token.text, 'lemma': token.lemma_} for token in entity_text]
        entity_texts.append(tokens)
    json_obj['entity_texts'] = entity_texts
    json_obj['linking_features'] = instance.fields['table'].linking_features
    return json.dumps(json_obj)


if __name__ == '__main__':
    param_file = sys.argv[1]
    outdir = 'wikitables_preprocessed_data/'
    params = Params.from_file(param_file)
    main(params, outdir)
