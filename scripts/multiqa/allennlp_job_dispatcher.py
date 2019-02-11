import argparse
import json
import pika
import datetime
import time
import boto3
import copy
import sys
import glob
sys.path.insert(0, '/Users/alontalmor/Dropbox/Backup/QAResearch/allennlp')
from allennlp.common.elastic_logger import ElasticLogger
config_path = '/Users/alontalmor/Dropbox/Backup/QAResearch/MultiQA/experiment_configs/'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config_file(filename):
    full_file = glob.glob(config_path + '*/*' + filename + '*' , recursive=True)[0]
    job_type = list(set(['train', 'evaluate','finetune']) & set(full_file.split('/')))[0]
    return full_file, job_type

def get_running_experiments():
    query = {
        "from" :0, "size" : 100,
        "query": {
            "bool": {
              "must": [
                { "match": { "message": "Job"}}
              ]
            }
        },
        "sort": [
            {
              "log_timestamp": {
                "order": "desc"
              }
            }
          ]
    }
    running_exp = []
    res = ElasticLogger().es.search(index="multiqa_logs", body=query)
    curr_time = datetime.datetime.utcnow()
    for exp in res['hits']['hits']:
        exp_time = datetime.datetime.strptime(exp['_source']['log_timestamp'],'%Y-%m-%dT%H:%M:%S.%f')
        if curr_time - exp_time < datetime.timedelta(0,180):
            running_exp.append(exp['_source']['experiment_name'])
    return list(set(running_exp))

def get_s3_experiments(prefix):
    s3 = boto3.client("s3")
    all_objects = s3.list_objects(Bucket='multiqa',Prefix=prefix)
    all_keys = []
    if 'Contents' in all_objects:
        for obj in all_objects['Contents']:
            if obj['Key'].find('.tar.gz') > -1:
                all_keys.append(obj['Key'])
    return all_keys

def allennlp_include_packages():
    return ' --include-package allennlp.models.reading_comprehension.docqa++ \
               --include-package allennlp.data.iterators.multiqa_iterator \
               --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa+ \
               --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa+combine \
               --include-package allennlp.models.reading_comprehension.docqa++BERT \
               --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa_bert'

def read_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def add_model_dir(host):
    if host in ['savant', 'rack-gamir-g03', 'rack-gamir-g04', 'rack-gamir-g05']:
        return '/home/joberant/home/alontalmor/models/'
    else:
        return '../models/'

def send_to_queue(name, queue, config):
    connection_params = pika.URLParameters('amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    config['retry'] = 0
    channel.basic_publish(exchange='',
                          properties=pika.BasicProperties(
                              headers={'name': name}),
                          routing_key=queue,
                          body=json.dumps(config))
    connection.close()

def dataset_specific_override(dataset, run_config):
    if dataset in run_config['dataset_specific_override']:
        for key1 in run_config['dataset_specific_override'][dataset].keys():
            for key2 in run_config['dataset_specific_override'][dataset][key1].keys():
                if type(run_config['dataset_specific_override'][dataset][key1][key2]) == dict:
                    for key3 in run_config['dataset_specific_override'][dataset][key1][key2].keys():
                        run_config['override_config'][key1][key2][key3] = \
                            run_config['dataset_specific_override'][dataset][key1][key2][key3]
                else:
                    run_config['override_config'][key1][key2] = \
                        run_config['dataset_specific_override'][dataset][key1][key2]

    return run_config

def replace_tags(dataset, run_config, source_dataset=None):
    run_config['override_config']["train_data_path"] = \
        run_config['override_config']["train_data_path"].replace('[DATASET]', dataset)
    run_config['override_config']["validation_data_path"] = \
        run_config['override_config']["validation_data_path"].replace('[DATASET]', dataset)

    run_config['output_file'] = run_config['output_file'].replace('[RUN_NAME]', run_name)
    run_config['output_file_cloud_target'] = run_config['output_file_cloud_target'].replace('[EXP_NAME]', experiment_name)
    run_config['output_file_cloud_target'] = run_config['output_file_cloud_target'].replace('[DATASET]', dataset)

    if 'source_model_path' in run_config:
        run_config['source_model_path'] = run_config['source_model_path'].replace('[SOURCE]', source_dataset)
        run_config['output_file'] = run_config['output_file'].replace('[SOURCE]', source_dataset)
        run_config['output_file_cloud_target'] = run_config['output_file_cloud_target'].replace('[SOURCE]', source_dataset)

    return run_config

def replace_tag_params(run_config, params):
    for key in params.keys():
        if "train_data_path" in run_config['override_config']:
            run_config['override_config']["train_data_path"] = \
                run_config['override_config']["train_data_path"].replace('[' + key + ']', str(params[key]))
        if "validation_data_path" in run_config['override_config']:
            run_config['override_config']["validation_data_path"] = \
                run_config['override_config']["validation_data_path"].replace('[' + key + ']', str(params[key]))

        if "post_proc_bash" in run_config:
            run_config["post_proc_bash"] = run_config["post_proc_bash"].replace('[' + key + ']', str(params[key]))

        if "model" in run_config:
            run_config["model"] = run_config["model"].replace('[' + key + ']', str(params[key]))
        if "eval_set" in run_config:
            run_config["eval_set"] = run_config["eval_set"].replace('[' + key + ']', str(params[key]))

        run_config['output_file'] = run_config['output_file'].replace('[' + key + ']', str(params[key]))
        if 'output_file_cloud' in run_config:
            run_config['output_file_cloud'] = \
                run_config['output_file_cloud'].replace('[' + key + ']', str(params[key]))
    return run_config

def replace_one_field_tags(value, params):
    for key in params.keys():
        value = value.replace('[' + key + ']', str(params[key]))

    return value

def build_experiments_params(config):
    experiments = []
    iterators = []
    for iterator in config['nested_iterators'].keys():
        expanded_experiments = []
        if len(experiments) > 0:
            for experiment in experiments:
                for value in config['nested_iterators'][iterator]:
                    new_expriment = copy.deepcopy(experiment)
                    new_expriment.update({iterator: value})
                    expanded_experiments.append(new_expriment)
        else:
            for value in config['nested_iterators'][iterator]:
                new_expriment = {iterator: value}
                expanded_experiments.append(new_expriment)
        experiments = expanded_experiments

    if len(config['list_iterators']) > 0:
        expanded_experiments = []
        for value in config['list_iterators']:
            if len(experiments) > 0:
                for experiment in experiments:
                    new_expriment = copy.deepcopy(experiment)
                    new_expriment.update(value)
                    expanded_experiments.append(new_expriment)
            else:
                new_expriment = value
                expanded_experiments.append(new_expriment)
        experiments = expanded_experiments
    return experiments

def build_evaluate_bash_command(run_config, run_name):
    bash_command = 'python -m allennlp.run evaluate ' + run_config['model'] + ' '
    bash_command += run_config['eval_set'] + ' '
    bash_command += '--output-file ' + run_config['output_file'] + ' '
    bash_command += '-o "' + str(run_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += ' --cuda-device [GPU_ID]'
    bash_command += allennlp_include_packages()
    return bash_command

def build_train_bash_command(run_config, run_name):
    bash_command = 'python -m allennlp.run train ' + run_config['master_config'] + ' '
    bash_command += '-s ' + '[MODEL_DIR]' + run_name + ' '
    bash_command += '-o "' + str(run_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += allennlp_include_packages()
    return bash_command

def build_finetune_bash_command(run_config, run_name):
    bash_command = 'python -m allennlp.run fine-tune -m ' + run_config['source_model_path'] + ' '
    bash_command += '-c ' + run_config['master_config'] + ' '
    bash_command += '-s ' + '[MODEL_DIR]' + run_name + ' '
    bash_command += '-o "' + str(run_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += allennlp_include_packages()
    return bash_command

def run_job(experiment_name, DRY_RUN):
    # queue = 'rack-jonathan-g02'
    # queue = 'rack-gamir-g03'
    FORCE_RUN = False
    queue = 'GPUs'
    print('Running new job on queue = %s', queue)

    full_file, job_type = get_config_file(experiment_name)
    if not DRY_RUN:
        print('------------ EXECUTING -------------')
        print('------------ EXECUTING -------------')
        time.sleep(5)

    config = read_config(full_file)
    print('Description: %s \n\n' % (config['description']))
    config['override_config'] = config['allennlp_override']

    s3_done_experiments = []
    if 'output_file_cloud' in config:
        s3_done_experiments = get_s3_experiments('/'.join(config['output_file_cloud'].split('/')[3:6]))
    currently_running_experiments = get_running_experiments()

    experiments = build_experiments_params(config)

    for params in experiments:
        run_config = copy.deepcopy(config)
        if 'TARGET' in params:
            run_config = dataset_specific_override(params['TARGET'], run_config)
        elif 'DATASET' in params:
            run_config = dataset_specific_override(params['DATASET'], run_config)

        params['EXPERIMENT'] = experiment_name
        # adding execution time name
        run_name_no_date = replace_one_field_tags(run_config['run_name'], params)
        run_name = run_name_no_date + '_' + datetime.datetime.now().strftime("%m%d_%H%M")
        params['RUN_NAME'] = run_name
        if 'output_file_cloud' in run_config:
            params['OUTPUT_FILE_CLOUD'] = replace_one_field_tags(run_config['output_file_cloud'], params)
        run_config = replace_tag_params(run_config, params)

        # checking if this run has already been run
        if not FORCE_RUN:
            if job_type == 'evaluate':  # evaluate stores results in elastic..
                pass
            else:
                if '/'.join(run_config['output_file_cloud'].split('/')[3:]) in s3_done_experiments:
                    print('!! %s already found in s3, NOT RUNNING. \n' % (run_name))
                    continue
            if len([exp for exp in currently_running_experiments if exp.lower().startswith(run_name_no_date.lower())]) > 0:
                print('!! %s currently running , NOT RUNNING. \n' % (run_name_no_date))
                continue

        # Building command
        runner_config = {'output_file': run_config['output_file']}
        runner_config['operation'] = "run job"

        if "post_proc_bash" in run_config:
            runner_config['post_proc_bash'] = run_config['post_proc_bash']
        runner_config['resource_type'] = 'GPU'
        runner_config['override_config'] = run_config['override_config']
        if job_type == 'evaluate':
            runner_config['bash_command'] = build_evaluate_bash_command(run_config, run_name)
        elif job_type == 'train':
            runner_config['bash_command'] = build_train_bash_command(run_config, run_name)
        elif job_type == 'finetune':
            runner_config['bash_command'] = build_finetune_bash_command(run_config, run_name)
        runner_config['bash_command'] = replace_one_field_tags(runner_config['bash_command'], params)
        print(run_name)
        print('## bash_command= %s' % (runner_config['bash_command'].replace(',', ',\n').replace(' -', '\n-')))
        if 'run_config' in run_config:
            print('## output_file_cloud= %s' % (run_config['output_file_cloud']))
        print('## output_file= %s' % (runner_config['output_file']))
        if "post_proc_bash" in run_config:
            print('## post_proc_bash= %s' % (runner_config['post_proc_bash']))
        print('--------------------------------\n')

        if not DRY_RUN:
            send_to_queue(run_name, queue, runner_config)
            time.sleep(1)
        # break



# experiment_name = '017_BERT_exp2_finetune_from_partial_exp'
#experiment_name = '018_BERT_exp2_finetune_from_full_exp'
#experiment_name = '016_GloVe_exp3_finetune_target_sizes'
#experiment_name = '019_BERT_finetune_Full'
#experiment_name = '011_BERT_exp1_finetune_small'
experiment_name = '020_BERT_exp1_evaluate'

parse = argparse.ArgumentParser("")
parse.add_argument("--DRY_RUN", type=str2bool, default=True, help="")
args = parse.parse_args()
run_job(experiment_name, args.DRY_RUN)

