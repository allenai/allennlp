import json
import pika
import datetime


def allennlp_include_packages():
    return ' --include-package allennlp.models.reading_comprehension.docqa++ \
               --include-package allennlp.data.iterators.multiqa_iterator \
               --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa+ \
               --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa+combine '

def add_override_config(config_name):
    config_path = '/Users/alontalmor/Dropbox/Backup/QAResearch/MultiQA/configs/'
    with open(config_path + config_name, 'r') as f:
         return json.load(f)

def add_model_dir(host):
    if host in ['savant', 'rack-gamir-g03', 'rack-gamir-g04', 'rack-gamir-g05']:
        return '/home/joberant/home/alontalmor/models/'
    else:
        return '../models/'

def dispatch():
    name = ''
    config = {}

    ## Operation ##
    #operation = "train"
    operation = "evaluate"
    #Operation = "Preprocess"
    #Operation = "Recover"
    #Operation = "kill job"
    #Operation = "git pull"
    #operation = "restart runner"
    train_schema = 'ELMo'
    
    ## HOST ##
    #host = 'rack-gamir-g03'
    #host = 'rack-gamir-g04'
    #host = 'rack-gamir-g04'
    #host = 'savant'
    #host = 'rack-jonathan-g02'

    if operation == "Preprocess":
        host = 'z-rack-jonathan-02'
    #host = 'z-rack-jonathan-02'
    host = 'test'

    if operation == "train":
        config['override_config'] = add_override_config('MultiQA_Override.json')
        dataset = 'NewsQA'
        config['override_config']['trainer']["cuda_device"] = None

        #config['override_config']["train_data_path"] =  "s3://multiqa/15000/" + dataset + "_15000_400_train.jsonl.zip"
        #config['override_config']["train_data_path"] =  "s3://multiqa/75000/" + dataset + "_75000_400_train.jsonl.zip"
        #config['override_config']["train_data_path"] = "s3://multiqa/preproc/" + dataset + "_400_train.jsonl.zip"
        config['override_config']["train_data_path"] = "s3://multiqa/preproc/" + dataset + "_1200_train.jsonl.zip"
        config['override_config']["validation_data_path"] = "s3://multiqa/preproc/" + dataset + "_400_dev.jsonl.zip"

        # Command
        config['operation'] = 'train'
        config['master_config'] = 's3://multiqa/config/MultiQA_ELMo.json '
        #config['master_config'] = 's3://multiqa/config/MultiQA_GloVe_no_char_tokens.json '
        #config['master_config'] = 's3://multiqa/config/MultiQA_GloVe.json '



        # Dataset name:
            # dataset combining (multi-task training)
        #config['override_config']['dataset_reader']["type"] = "multiqa+combine"
        #config['override_config']["train_data_path"] = "s3://multiqa/75000/NewsQA_75000_400_train.jsonl.zip,s3://multiqa/15000/Squad_15000_400_train.jsonl.zip"
        #name = 'Squad_15000+NewsQA_75000/'

        name = config['override_config']['train_data_path'].split('/')[-1].replace('_train.json.zip', '').replace('_train.jsonl.zip', '') + '/'
        if config['master_config'] == 's3://multiqa/config/MultiQA_ELMo.json ':
            name += 'ELMo_'
        else:
            name += 'GloVe_'
        #name += 'no_sharednorm_'
        #name += '10docs_250tokens_'
        #name += 'lazy_' + str(dataset_specific_config['dataset_reader']['lazy']) + '_'
        #name += str(dataset_specific_config['trainer']['optimizer']['type']) + '_'
        #name += 'GPU' + str(config['override_config']['trainer']['cuda_device']) + '_'
        #name += 'LR_' + str(dataset_specific_config['trainer']['optimizer']['lr']) + '_'
        #name += 'L2_' + str(dataset_specific_config['trainer']['optimizer']['weight_decay'])  + '_'
        #name += 'shared_' + str(dataset_specific_config['model']['shared_norm'])  + '_'
        name += datetime.datetime.now().strftime("%m%d_%H%M")

        # building bash command
        bash_command = 'python -m allennlp.run train ' + config['master_config']
        bash_command += '--s ' + config['model_dir'] + name + ' '
        # Building the python command with arguments
        bash_command += '-o "' + str(config['override_config']).replace('True', 'true').replace('False', 'false') + '"'
        bash_command += allennlp_include_packages()

    elif operation == "evaluate":

        base_model = 's3://multiqa/models/75000/Squad.tar.gz '
        tar_set = 's3://multiqa/preproc/Squad_400_dev.jsonl.zip '
        config['override_config'] = add_override_config('MultiQA_Override.json')
        config['master_config'] = 's3://multiqa/config/MultiQA_GloVe.json '
        # todo add name
        name = 'eval_Squad_dev_on_Squad_75000'
        config['operation'] = "run job"
        bash_command = 'python -m allennlp.run evaluate ' + base_model  + tar_set
        bash_command += '--output-file ' + name + '.json '
        bash_command += allennlp_include_packages()
        bash_command += ' --cuda-device [GPU_ID]'

        config['post_proc_bash'] = 'aws s3 cp ' + name + '.json s3://multiqa/eval/' + name + '.json --acl public-read'
        config['resource_type'] = 'GPU'
        config['bash_command'] = bash_command

    # preprocess
    elif operation == "Preprocess":
        set = 'dev'
        dataset = 'TriviaQA-web'
        docsize = '250'
        name = "preprocess_" + dataset + "_" + docsize + "_" + set + "_" + datetime.datetime.now().strftime("%m%d_%H%M")
        command = "python scripts/multiqa/preprocess.py s3://multiqa/datasets/" + dataset + "_" + set + ".jsonl.zip s3://multiqa/preproc/" + dataset \
                  + "_" + docsize + "_" + set + ".jsonl.zip --n_processes 10 --ndocs 10 --docsize " + docsize + \
                  " --titles True --use_rank True --require_answer_in_doc False --sample_size -1  "
        if set == "train":
            command += "--require_answer_in_question True"
        else:
            command += "--require_answer_in_question False"

    elif operation == "kill job":
        name = "kill job"
        config['operation'] = "kill job"
        config['experiment_name'] = "HotpotQA_75000_400/GloVe_GPU3_0123_2137"

    elif operation == "restart runner":
        config['operation'] = "restart runner"

    print('\n')
    print(host)
    print(name)
    print(config)
    
    connection_params = pika.URLParameters('amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    channel.basic_publish(exchange='',
                        properties=pika.BasicProperties(
                            headers={'name':name}),
                        routing_key=host,
                        body=json.dumps(config))
    connection.close()

dispatch()

