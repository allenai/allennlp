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
    full_file = glob.glob(config_path + '*/*' + filename + '*' , recursive=True)
    if len(full_file) == 0:
        full_file = glob.glob(config_path + '*/*/*' + filename + '*', recursive=True)
        if len(full_file) == 0:
            full_file = glob.glob(config_path + '*/*/*/*' + filename + '*', recursive=True)
    return full_file[0]

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

def dataset_specific_override(dataset, exp_config):
    if dataset in exp_config['dataset_specific_override']:
        for key1 in exp_config['dataset_specific_override'][dataset].keys():
            for key2 in exp_config['dataset_specific_override'][dataset][key1].keys():
                if type(exp_config['dataset_specific_override'][dataset][key1][key2]) == dict:
                    for key3 in exp_config['dataset_specific_override'][dataset][key1][key2].keys():
                        exp_config['override_config'][key1][key2][key3] = \
                            exp_config['dataset_specific_override'][dataset][key1][key2][key3]
                else:
                    exp_config['override_config'][key1][key2] = \
                        exp_config['dataset_specific_override'][dataset][key1][key2]

    return exp_config

def replace_tags(dataset, exp_config, source_dataset=None):
    exp_config['override_config']["train_data_path"] = \
        exp_config['override_config']["train_data_path"].replace('[DATASET]', dataset)
    exp_config['override_config']["validation_data_path"] = \
        exp_config['override_config']["validation_data_path"].replace('[DATASET]', dataset)

    exp_config['output_file'] = exp_config['output_file'].replace('[RUN_NAME]', run_name)
    exp_config['output_file_cloud_target'] = exp_config['output_file_cloud_target'].replace('[EXP_NAME]', experiment_name)
    exp_config['output_file_cloud_target'] = exp_config['output_file_cloud_target'].replace('[DATASET]', dataset)

    if 'source_model_path' in exp_config:
        exp_config['source_model_path'] = exp_config['source_model_path'].replace('[SOURCE]', source_dataset)
        exp_config['output_file'] = exp_config['output_file'].replace('[SOURCE]', source_dataset)
        exp_config['output_file_cloud_target'] = exp_config['output_file_cloud_target'].replace('[SOURCE]', source_dataset)

    return exp_config

def replace_tag_params(exp_config, params):
    for key in params.keys():
        if "train_data_path" in exp_config['override_config']:
            exp_config['override_config']["train_data_path"] = \
                exp_config['override_config']["train_data_path"].replace('[' + key + ']', str(params[key]))
        if "validation_data_path" in exp_config['override_config']:
            exp_config['override_config']["validation_data_path"] = \
                exp_config['override_config']["validation_data_path"].replace('[' + key + ']', str(params[key]))
        if 'condition' in exp_config:
            exp_config["condition"] = exp_config["condition"].replace('[' + key + ']', str(params[key]))
        if "post_proc_bash" in exp_config:
            exp_config["post_proc_bash"] = exp_config["post_proc_bash"].replace('[' + key + ']', str(params[key]))

        if "model" in exp_config:
            exp_config["model"] = exp_config["model"].replace('[' + key + ']', str(params[key]))
        if "eval_set" in exp_config:
            exp_config["eval_set"] = exp_config["eval_set"].replace('[' + key + ']', str(params[key]))

        exp_config['output_file'] = exp_config['output_file'].replace('[' + key + ']', str(params[key]))
        if 'output_file_cloud' in exp_config:
            exp_config['output_file_cloud'] = \
                exp_config['output_file_cloud'].replace('[' + key + ']', str(params[key]))
    return exp_config

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

def build_evaluate_bash_command(exp_config, run_name):
    bash_command = 'python -m allennlp.run evaluate ' + exp_config['model'] + ' '
    bash_command += exp_config['eval_set'] + ' '
    bash_command += '--output-file ' + exp_config['output_file'] + ' '
    bash_command += '-o "' + str(exp_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += ' --cuda-device [GPU_ID]'
    bash_command += allennlp_include_packages()
    return bash_command

def build_train_bash_command(exp_config, run_name):
    bash_command = 'python -m allennlp.run train ' + exp_config['master_config'] + ' '
    bash_command += '-s ' + '[MODEL_DIR]' + run_name + ' '
    bash_command += '-o "' + str(exp_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += allennlp_include_packages()
    return bash_command

def build_finetune_bash_command(exp_config, run_name):
    bash_command = 'python -m allennlp.run fine-tune -m ' + exp_config['source_model_path'] + ' '
    bash_command += '-c ' + exp_config['master_config'] + ' '
    bash_command += '-s ' + '[MODEL_DIR]' + run_name + ' '
    bash_command += '-o "' + str(exp_config['override_config']).replace('True', 'true').replace('False', 'false') + '" '
    bash_command += allennlp_include_packages()
    return bash_command

def get_elastic_evaluate_exp_results(experiment_name):
    query = {
        "from": 0, "size": 500,
        "query": {
            "bool": {
                "must": [
                    {"match": {"experiment": experiment_name}}],
            }
        }
    }
    res = ElasticLogger().es.search(index="multiqa_logs", body=query)
    return [r['_source'] for r in res['hits']['hits']]

def check_if_job_exists(exp_config, job_type, s3_done_experiments, \
                        currently_running_experiments, run_name, run_name_no_date, elastic_exp_results, params):
    if job_type == 'evaluate':  # evaluate stores results in elastic..
        # TODO add full experiment name to results and just match, this is ugly...
        if 'EVAL_SET' in params:
            elastic_exp_results = [res for res in elastic_exp_results if res['eval_set'] == params['EVAL_SET']]
        else:
            elastic_exp_results = [res for res in elastic_exp_results if res['eval_set'] == params['TARGET']]

        if 'TARGET' in params:
            elastic_exp_results = [res for res in elastic_exp_results if res['target_dataset'] == params['TARGET']]
        else:
            elastic_exp_results = [res for res in elastic_exp_results if res['target_dataset'] == params['MODEL']]

        if 'SOURCE' in params:
            elastic_exp_results = [res for res in elastic_exp_results if res['source_dataset'] == params['SOURCE']]

        if 'TARGET_SIZE' in params:
            elastic_exp_results = [res for res in elastic_exp_results if res['target_size'] == params['TARGET_SIZE']]
        if len(elastic_exp_results) > 0:
            print('!! %s already found in elastic, NOT RUNNING. \n' % (run_name))
            return True

    else:
        if '/'.join(exp_config['output_file_cloud'].split('/')[3:]) in s3_done_experiments:
            print('!! %s already found in s3, NOT RUNNING. \n' % (run_name))
            return True
    if len([exp for exp in currently_running_experiments if exp.lower().startswith(run_name_no_date.lower())]) > 0:
        print('!! %s currently running , NOT RUNNING. \n' % (run_name_no_date))
        return True
    return False

def check_s3_resource_exists(resource):
    s3 = boto3.client("s3")
    prefix = '/'.join(resource.split('/')[3:])
    all_objects = s3.list_objects(Bucket='multiqa', Prefix=prefix)
    if 'Contents' in all_objects and all_objects['Contents'][0]['Key'] == prefix:
        return True
    else:
        return False



def run_job(experiment_name, DRY_RUN, queue, FORCE_RUN):


    full_file = get_config_file(experiment_name)

    if not DRY_RUN:
        print('------------ EXECUTING -------------')
        print('------------ EXECUTING -------------')
        time.sleep(5)

    config = read_config(full_file)
    job_type = config['operation']
    print('Description: %s \n\n' % (config['description']))
    config['override_config'] = config['allennlp_override']

    s3_done_experiments = []
    if 'output_file_cloud' in config:
        s3_done_experiments = get_s3_experiments('/'.join(config['output_file_cloud'].split('/')[3:6]))
    currently_running_experiments = get_running_experiments()
    elastic_exp_results = []
    if job_type == 'evaluate':
        elastic_exp_results = get_elastic_evaluate_exp_results(experiment_name)

    experiments = build_experiments_params(config)

    runnable_count = 0
    for params in experiments:
        exp_config = copy.deepcopy(config)
        if 'TARGET' in params:
            exp_config = dataset_specific_override(params['TARGET'], exp_config)
        elif 'DATASET' in params:
            exp_config = dataset_specific_override(params['DATASET'], exp_config)

        params['EXPERIMENT'] = experiment_name
        # adding execution time name
        run_name_no_date = replace_one_field_tags(exp_config['run_name'], params)
        run_name = run_name_no_date + '_' + datetime.datetime.now().strftime("%m%d_%H%M")
        params['RUN_NAME'] = run_name
        if 'output_file_cloud' in exp_config:
            params['OUTPUT_FILE_CLOUD'] = replace_one_field_tags(exp_config['output_file_cloud'], params)
        exp_config = replace_tag_params(exp_config, params)

        if 'condition' in exp_config and not eval(exp_config['condition']):
            continue

        if 'model' in exp_config and not check_s3_resource_exists(exp_config['model']):
            print('%s s3 resource does not exist, skipping\n' % exp_config['model'])
        if 'eval_set' in exp_config and not check_s3_resource_exists(exp_config['model']):
            print('%s s3 resource does not exist, skipping\n' % exp_config['eval_set'])


        # checking if this run has already been run
        if not FORCE_RUN:
            if check_if_job_exists(exp_config, job_type, s3_done_experiments, \
                    currently_running_experiments, run_name, run_name_no_date, elastic_exp_results, params):
                continue


        # Building command
        runner_config = {'output_file': exp_config['output_file']}
        runner_config['operation'] = "run job"

        if "post_proc_bash" in exp_config:
            runner_config['post_proc_bash'] = exp_config['post_proc_bash']
        runner_config['resource_type'] = 'GPU'
        runner_config['override_config'] = exp_config['override_config']
        if job_type == 'evaluate':
            runner_config['bash_command'] = build_evaluate_bash_command(exp_config, run_name)
        elif job_type == 'train':
            runner_config['bash_command'] = build_train_bash_command(exp_config, run_name)
        elif job_type == 'finetune':
            runner_config['bash_command'] = build_finetune_bash_command(exp_config, run_name)
        runner_config['bash_command'] = replace_one_field_tags(runner_config['bash_command'], params)
        print(run_name)
        print('## bash_command= %s' % (runner_config['bash_command'].replace(',', ',\n').replace(' -', '\n-')))
        if 'output_file_cloud' in exp_config:
            print('## output_file_cloud= %s' % (exp_config['output_file_cloud']))
        print('## output_file= %s' % (runner_config['output_file']))
        if "post_proc_bash" in exp_config:
            print('## post_proc_bash= %s' % (runner_config['post_proc_bash']))
        print('--------------------------------\n')
        runnable_count += 1
        if not DRY_RUN:
            send_to_queue(run_name, queue, runner_config)
            time.sleep(1)
        break
    print('%d runnable experiments' % (runnable_count))




#experiment_name = '017_BERT_exp2_finetune_from_partial_exp'
#experiment_name = '018_BERT_exp2_finetune_from_full_exp'
#experiment_name = '016_GloVe_exp3_finetune_target_sizes'
#experiment_name = '019_BERT_finetune_Full'
#experiment_name = '011_BERT_exp1_finetune_small'
#experiment_name = '020_BERT_exp1_evaluate'
#experiment_name = '021_BERT_exp2_finetune_small_from_75K_exp'
#experiment_name = '022_DocQA_exp1_evaluate'
#experiment_name = '023_DocQA_exp2_evaluate'
experiment_name = '024_BERT_exp2_evaluate'
#experiment_name = '025_GloVe_exp3_finetune_target_sizes_lr'

#queue = 'rack-jonathan-g02'
# queue = 'rack-gamir-g03'
queue = 'GPUs'
if queue != 'GPUs':
    print('!!!!!!!!!!! RUNNING ON A SPECIFIC MACHINE, ARE YOU SURE?? !!!!!!!!')
    time.sleep(5)

FORCE_RUN = False


print('Running new job on queue = %s', queue)
parse = argparse.ArgumentParser("")
parse.add_argument("--DRY_RUN", type=str2bool, default=True, help="")
args = parse.parse_args()
run_job(experiment_name, args.DRY_RUN , queue, FORCE_RUN)

