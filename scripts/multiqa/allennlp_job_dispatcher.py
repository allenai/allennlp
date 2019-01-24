import json
import pika
import datetime

def dispatch():

    Operation = "train"
    Operation = "Preprocess"
    #Operation = "Recover"
    #Operation = "kill job"
    #Operation = "git pull"
    
    ## HOST
    #host = 'rack-gamir-g03'
    #host = 'rack-gamir-g05'
    #host = 'rack-gamir-g04'
    #host = 'savant'
    host = 'rack-jonathan-g04'

    if Operation == "Preprocess":
        host = 'z-rack-jonathan-02'
        #host = 'rack-jonathan-02'
    #host = 'z-rack-jonathan-02'

    # host = 'test'

    if host in ['savant', 'rack-gamir-g03', 'rack-gamir-g04', 'rack-gamir-g05']:
        model_dir = '/home/joberant/home/alontalmor/models/'
    else:
        model_dir = '../models/'

    # CONFIG PARAMS

    config_path = '/Users/alontalmor/Dropbox/Backup/QAResearch/MultiQA/configs/'
    #config_json = 'SearchQA.json'
    #config_json = 'Squad.json'
    #config_json = 'NewsQA.json'
    #config_json = 'HotpotQA.json'
    config_json = 'ComplexWebQuestions.json'
    #config_json = 'TriviaQA-G.json'
    #config_json = 'TriviaQA-web.json'
    with open(config_path + config_json, 'r') as f:
        dataset_specific_config = json.load(f)

    # Dataset name:
    name = dataset_specific_config['train_data_path'].split('/')[-1].replace('_train.json.zip', '').replace('_train.jsonl.zip', '') + '/'
    name += 'GloVe_'
    #name += 'no_sort'
    name += 'patiente_12_20_all_q_in_train'
    #name += '10docs_250tokens_'
    #name += 'lazy_' + str(dataset_specific_config['dataset_reader']['lazy']) + '_'
    #name += str(dataset_specific_config['trainer']['optimizer']['type']) + '_'
    name += 'GPU' + str(dataset_specific_config['trainer']['cuda_device']) + '_'
    #name += 'LR_' + str(dataset_specific_config['trainer']['optimizer']['lr']) + '_'
    #name += 'L2_' + str(dataset_specific_config['trainer']['optimizer']['weight_decay'])  + '_'
    #name += 'shared_' + str(dataset_specific_config['model']['shared_norm'])  + '_'
    name += datetime.datetime.now().strftime("%m%d_%H%M")

    # Command
    command = 'python -m allennlp.run train ' 
    #command = 'python -m allennlp.run make-vocab ' 
    #command = 'python -m allennlp.run dry-run '
    # base configuration file (mainly for model and tokenization)
    #command += 's3://multiqa/config/MultiQA_GloVe.json '
    command += 's3://multiqa/config/MultiQA_GloVe_no_char_tokens.json '

    command += '--s ' + model_dir + name + ' '

    # Building the python command with arguments
    command += '-o "' + str(dataset_specific_config).replace('True', 'true').replace('False', 'false') + '"' 


    ## RECOVERS
    if Operation == "Recover":
        name = 'Squad_1000/GloVe_shared_False_0113_0815'
        command = '''python -m allennlp.run train s3://multiqa/config/MultiQA_GloVe_rand_iter.json --s ../models/Squad_1000/GloVe_shared_False_0113_0815 -o "{'dataset_reader': {'type': 'multiqa+', 'lazy': false}, 'iterator': {'type': 'multiqa', 'batch_size': 130, 'max_instances_in_memory': 10000, 'sorting_keys': [['passage', 'num_tokens'], ['question', 'num_tokens']]}, 'model': {'type': 'docqa++', 'frac_of_validation_used': 1.0, 'frac_of_training_used': 1.0, 'shared_norm': false}, 'trainer': {'cuda_device': 1, 'learning_rate_scheduler': {'type': 'reduce_on_plateau', 'factor': 0.4, 'mode': 'max', 'patience': 5}, 'num_epochs': 80, 'optimizer': {'type': 'adam', 'lr': 0.001, 'weight_decay': 0.0001}, 'patience': 10, 'validation_metric': '+f1'}, 'validation_iterator': {'type': 'multiqa', 'batch_size': 80, 'max_instances_in_memory': 10000, 'sorting_keys': [['passage', 'num_tokens'], ['question', 'num_tokens']]}, 'train_data_path': 's3://multiqa/preproc/Squad_1000_train.json.zip', 'validation_data_path': 's3://multiqa/preproc/Squad_1000_dev.json.zip'}"''' + " --recover"

     # Packages
    command += ' --include-package allennlp.models.reading_comprehension.docqa++ \
        --include-package allennlp.data.iterators.multiqa_iterator \
        --include-package allennlp.data.dataset_readers.reading_comprehension.multiqa+ '
    #command += ' -f'

    # preprocess
    if Operation == "Preprocess":
        set = 'train'
        dataset = 'TriviaQA-web'
        docsize = '400'
        sample = '15000'
        if sample != '-1':
            name = "preprocess_" + dataset + "_" + docsize + "_" + sample + "_" + set + "_" + datetime.datetime.now().strftime("%m%d_%H%M")
            command = "python scripts/multiqa/preprocess.py s3://multiqa/datasets/" + dataset + "_" + set + ".jsonl.zip s3://multiqa/" \
                      + sample + "/" + dataset   + "_" + sample + "_" + docsize + "_" + set + ".jsonl.zip --n_processes 7 --ndocs 10 --docsize " \
                      + docsize + " --titles True --use_rank True --require_answer_in_doc False --sample_size " + sample + " "
        else:
            name = "preprocess_" + dataset + "_" + docsize + "_" + set + "_" + datetime.datetime.now().strftime("%m%d_%H%M")
            command = "python scripts/multiqa/preprocess.py s3://multiqa/datasets/" + dataset + "_" + set + ".jsonl.zip s3://multiqa/preproc/" \
                      + dataset + "_" + docsize + "_" + set + ".jsonl.zip --n_processes 10 --ndocs 10 --docsize " + docsize + \
                      " --titles True --use_rank True --require_answer_in_doc False --sample_size " + sample + " "

        if set == "train":
            command += "--require_answer_in_question True"
        else:
            command += "--require_answer_in_question False"

    if Operation == "git pull":
        name = "gitpull"
        command = "git pull origin master"
    
    if Operation == "kill job":
        name = "kill job"
        command = "ComplexWebQuestions_250/GloVe_GPU1_0120_1447"
    
    print('\n')
    print(host)
    print(name)
    print(command)
    
    connection_params = pika.URLParameters('amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    channel.basic_publish(exchange='',
                        properties=pika.BasicProperties(
                            headers={'name':name}),
                        routing_key=host,
                        body=command)
    connection.close()

dispatch()

