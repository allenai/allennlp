
# TODO change...
RUNNER_VERSION = "0.2.0"
from allennlp.common.util import *
from subprocess import Popen,call
import traceback,os, psutil
import time
import shutil
import logging
import argparse
import pickle
from allennlp.common.elastic_logger import ElasticLogger
import pika

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class JobRunner():
    def __init__(self, channel, type, models_dir, resources_to_spare, DEBUG, SIM_GPUS, job_no_run_string):
        self._SIM_GPUS = SIM_GPUS
        self._DEBUG = DEBUG
        self._job_no_run_string = job_no_run_string
        if self._DEBUG:  # pylint: disable=invalid-name
            level = logging.getLevelName('DEBUG')
            logger.setLevel(level)
        self._MODELS_DIR = models_dir
        self.resource_type = type
        self.running_jobs = []
        self.pending_jobs = []
        self.job_gpus = []
        self.log_handles = {}
        self.available_gpus = []
        self.channel_tags = channel
        self.connection = None
        self.channel = None
        self.resources_available = True
        self.connect_to_queue()
        self.runner_iter_count = 0
        self.resources_to_spare = resources_to_spare
        self.update_available_gpus()
        self.last_exception = None
        self.runned_job_names = []

    def update_available_gpus(self):
        if self._SIM_GPUS: # simulating gpu resources
            self.available_gpus = list(set([1,2]) - set(self.job_gpus))
            return

        try:
            gpu_mem = gpu_memory_mb()
            if self.channel == 'rack-jonathan-g02':
                gpu_mem = {(3 - key): val for key, val in gpu_mem.items()}

            self.available_gpus = [i for i, gpu in enumerate(gpu_mem.keys()) if
                                    gpu_mem[gpu] >= 300 and gpu_mem[gpu] < 1000 and i not in self.job_gpus]
            self.available_gpus += [i for i, gpu in enumerate(gpu_mem.keys()) if gpu_mem[gpu] < 300 and i not in self.job_gpus]

        except:
            self.available_gpus = []

    def inspect_job(self,job):
        logger.info('Inspecting job: %s' % (job['experiment_name']))
        # check if process still alive:
        try:
            logger.info('checcking if process alive ' + str(job['pid']))
            os.killpg(os.getpgid(job['pid']), 0)
        except:
            logger.info('killpg check not alive')
            job['alive'] = False

        try:
            # Log snapshot
            log_data = self.log_handles[job['log_file']].readlines()
            job['log_snapshot'] += ' '.join(log_data)
        except:
            logger.info('log read failed')
            job['alive'] = False

        # Log time out handling TODO
        try:
            statbuf = os.stat(job['log_file'])
            job['log_update_diff'] = time.time() - statbuf.st_mtime
        except:
            logger.info('file not found')

        return job

    def reopen_all_logs(self):
        for job in self.running_jobs:
            self.log_handles[job['log_file']] = open(job['log_file'], 'r')

    def close_all_logs(self):
        for log_name in self.log_handles:
            self.log_handles[log_name].close()
        self.log_handles = {}

    def close_job_log(self,job):
        # printing longer log
        if job['log_file'] in self.log_handles:
            self.log_handles[job['log_file']].close()
            with open(job['log_file'], 'r') as f:
                log_data = f.readlines()
            if len(log_data) > 100:
                job['log_snapshot'] = ' '.join(log_data[-100:])

    def handle_job_stopped(self,job):
        logger.info('processed not alive.')
        # checking if process successfully completed task,


        # reading the whole log in this case:
        if job['log_file'] in self.log_handles:
            self.log_handles[job['log_file']].seek(0)
            job['log_snapshot'] = self.log_handles[job['log_file']].read()



        if type(job['GPU']) == list:
            for GPU in job['GPU']:
                if GPU in self.job_gpus:
                    self.job_gpus.remove(GPU)
        else:
            if job['GPU'] in self.job_gpus:
                self.job_gpus.remove(job['GPU'])
        self.update_available_gpus()

        # TODO this is an ugly check for error , but because we are forking with nohup, python does not provide any good alternative...
        # We also assume here that jobs don't take less than 20 seconds...
        if ('output_file' in job and job['output_file'] is not None and not os.path.exists(job['output_file'])):

            if len(job['log_snapshot']) > 10001:
                job['log_snapshot'] = job['log_snapshot'][-10000:]
            # checking retries:

            if job['retries'] < 3 and job['command'].find(' train ') > -1:
                ElasticLogger().write_log('INFO', "Job Retry", {'experiment_name': job['experiment_name'], \
                        'command':job['command'].replace('&',' -f &'), \
                        'log_snapshot': job['log_snapshot']}, push_bulk=True, print_log=False)
                # old .replace('&',' --recover &')
                job['retries'] += 1
                # rerunning job:
                with open(job['log_file'], 'wb') as f:
                    if self._DEBUG:
                        wa_proc = Popen("nohup python dummy_job.py &", shell=True, preexec_fn=os.setsid, stdout=f, stderr=f)
                    else:
                        wa_proc = Popen(job['command'],\
                                        shell=True,preexec_fn=os.setsid, stdout=f, stderr=f)
                job['alive'] = True
                job['pid'] = wa_proc.pid + 1
                job['start_time'] =  time.time()
                return
            else:

                self.close_job_log(job)
                logger.info("resending to job %s to queue",job['experiment_name'])

                if job['config']['retry'] < 3 and job['channel'] == 'GPUs':
                    ElasticLogger().write_log('INFO', "Job Resend to Queue", {'experiment_name': job['experiment_name'],
                                                                    'log_snapshot': job['log_snapshot']}, push_bulk=True, print_log=False)
                    job['config']['retry'] += 1
                    # routing job to the GPUs
                    self.channel.basic_publish(exchange='',
                                      properties=pika.BasicProperties(
                                          headers={'name': job['experiment_name']}),
                                      routing_key='GPUs',
                                      body=json.dumps(job['config']))
                else:
                    ElasticLogger().write_log('INFO', "Job died", {'experiment_name': job['experiment_name'],
                                                                   'log_snapshot': job['log_snapshot']}, push_bulk=True, print_log=False)



        else:

            # running post proc job
            if 'post_proc_bash' in job['config'] and not job['is_post_proc_run']:
                post_proc_bash = job['config']['post_proc_bash']
                post_proc_bash = post_proc_bash.replace('[MODEL_DIR]', self._MODELS_DIR)
                logger.info('running post proc: %s',post_proc_bash)
                with open(job['log_file'], 'ab') as f:
                    wa_proc = Popen(post_proc_bash, shell=True , preexec_fn = os.setsid, stdout=f, stderr=f)
                job['pid'] = wa_proc.pid + 1
                job['alive'] = True
                job['is_post_proc_run'] = True
                return
            else:
                self.close_job_log(job)
                ElasticLogger().write_log('INFO', "Job finished successfully", {'experiment_name': job['experiment_name'],
                                                           'command': job['command'],\
                                                            'log_snapshot': job['log_snapshot']}, push_bulk=True, print_log=False)

        self.running_jobs.remove(job)
        if job['log_file'] in self.log_handles:
            self.log_handles.pop(job['log_file'])

    def write_status(self):
        # Virtual memory usage
        # Giving info on the processes running and GPU status

        # resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        for job in self.running_jobs:
            try:
                job['memory'] = psutil.Process(job['pid']).memory_info()
                # job['command'] = ' '.join(psutil.Process(wa_proc.pid+1).cmdline())
            except:
                job['alive'] = False

            ElasticLogger().write_log('INFO', "Job Status", {'experiment_name': job['experiment_name'],
                                                             'log_snapshot': job['log_snapshot']}, push_bulk=True, print_log=False)
            # Keeping last 3 log lines
            if len(job['log_snapshot']) > 0:
                job['log_snapshot'] = '\n'.join(job['log_snapshot'].split('\n')[-4:])
            else:
                job['log_snapshot'] = ''

        self.update_available_gpus()

        ElasticLogger().write_log('INFO', "Machine Status", {'resources_to_spare':self.resources_to_spare,\
                                                             'free_gpus': self.available_gpus, \
                                                             'num_procs_running': len(self.running_jobs), }, push_bulk=True, print_log=True)

    def connect_to_queue(self):
        #closing existing connection if any
        self.close_existing_connection()

        try:
            logger.info('reconnecting to queue')
            connection_params = pika.URLParameters(
                'amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
            self.connection = pika.BlockingConnection(connection_params)
            self.channel = self.connection.channel()
            logger.info('reconnected')
        except:
            logger.error('Error connecting: %s', traceback.format_exc())

    def close_existing_connection(self):
        try:
            if self.channel is not None:
                self.channel.close()
                self.channel = None
            if self.connection is not None:
                self.connection.close()
                self.connection = None
        except:
            logger.error('Error closing connection: %s', traceback.format_exc())
            ElasticLogger().write_log('INFO', "job runner exception", {'error_message': traceback.format_exc()}, print_log=True)

    def execute_job(self,name, bash_command, config, assigned_GPU, channel):
        # Creating the log dir
        log_file = 'logs/' + name + '.txt'
        log_dir_part = log_file.split('/')
        for dir_depth in range(len(log_dir_part) - 1):
            if not os.path.isdir('/'.join(log_dir_part[0:dir_depth + 1])):
                os.mkdir('/'.join(log_dir_part[0:dir_depth + 1]))

        # replace MARCOs
        bash_command = bash_command.replace('[MODEL_DIR]',self._MODELS_DIR)
        if not self._SIM_GPUS:
            bash_command = bash_command.replace("'[GPU_ID]'", str(assigned_GPU)).replace('[GPU_ID]', str(assigned_GPU))
            bash_command = bash_command.replace("'[GPU_ID4]'", str(assigned_GPU)).replace('[GPU_ID4]', str(assigned_GPU))

        else:
            bash_command = bash_command.replace("'[GPU_ID]'", str(assigned_GPU)).replace('[GPU_ID]', str(assigned_GPU))
            bash_command = bash_command.replace("'[GPU_ID4]'", str(assigned_GPU)).replace('[GPU_ID4]', str(assigned_GPU))

        if 'env_setup' in config:
            bash_command = config['env_setup'] + ' nohup ' + bash_command + ' &'
            proc_offset = len([c for c in config['env_setup'] if c == ';'])
        else:
            bash_command = 'nohup ' + bash_command + ' &'
            proc_offset = 0

        if 'output_file' in config and config['output_file'] is not None:
            config['output_file'] = config['output_file'].replace('[MODEL_DIR]', self._MODELS_DIR)

        # Executing
        logger.info(bash_command)
        with open(log_file, 'wb') as f:
            if self._DEBUG:
                wa_proc = Popen("nohup python dummy_job.py &", shell=True, preexec_fn=os.setsid, stdout=f, stderr=f)
            else:
                if config['retry'] > 0 and os.path.isdir(self._MODELS_DIR + name):
                    if bash_command.find(' train ')>-1:
                        wa_proc = Popen(bash_command.replace(' &',' -f &'), shell=True, preexec_fn=os.setsid, stdout=f, stderr=f)
                    else:
                        shutil.rmtree(self._MODELS_DIR + name)
                        wa_proc = Popen(bash_command, shell=True, preexec_fn=os.setsid, stdout=f, stderr=f)
                else:
                    wa_proc = Popen(bash_command, shell=True, preexec_fn=os.setsid, stdout=f, stderr=f)

                # lets give this some time...
                if not self._DEBUG:
                    time.sleep(10)


        # open log file for reading
        self.log_handles[log_file] = open(log_file, 'r')
        self.runned_job_names.append(name)
        new_job = {'GPU':assigned_GPU,'config': config,'command': bash_command, 'channel':channel, \
                    'log_file': log_file, 'log_snapshot': '', 'is_post_proc_run':False, \
                    'experiment_name': name, 'alive': True,'retries': 0, \
                    'pid': wa_proc.pid + 1 + proc_offset, 'start_time': time.time()}
        if 'output_file' in config:
            new_job['output_file'] =  config['output_file']
        self.running_jobs.append(new_job)
        self.update_available_gpus()

        ElasticLogger().write_log('INFO', "Job Started", {'GPU':assigned_GPU, 'command': bash_command, \
                    'experiment_name': name}, push_bulk=True, print_log=False)
        time.sleep(5)

    def handle_job_types(self, config, name, channel):
        if config['operation'] == 'run job':
            # Allocating resources
            assigned_GPU = -1
            if config['resource_type'] == 'GPU':
                if config['bash_command'].find('[GPU_ID4]') > -1:
                    assigned_GPU = []
                    for i in range(4):
                        single_assigned_GPU = self.available_gpus[-1]
                        self.job_gpus.append(single_assigned_GPU)
                        self.update_available_gpus()
                        logger.info('assigned_GPU = %s', single_assigned_GPU)
                        assigned_GPU.append(single_assigned_GPU)
                else:
                    if config['bash_command'].find('[GPU_ID]') > -1:
                        assigned_GPU = self.available_gpus[-1]
                    self.job_gpus.append(assigned_GPU)
                    self.update_available_gpus()
                    logger.info('assigned_GPU = %s', assigned_GPU)

            elif self._SIM_GPUS:
                assigned_GPU = self.available_gpus[-1]
                self.job_gpus.append(assigned_GPU)
                self.update_available_gpus()

            self.execute_job(name, config['bash_command'], config, assigned_GPU, channel)
        elif config['operation'] == 'kill job':
            logger.info('kill job!')
            job_to_kill = [job for job in self.running_jobs if job['experiment_name'] == config['experiment_name']]
            if len(job_to_kill) == 0:
                logger.info('job not found! %s ', config['experiment_name'])
            else:
                job_to_kill = job_to_kill[0]
            bash_command = 'kill ' + str(job_to_kill['pid'])
            proc_info = Popen(bash_command, shell=True)
            self.close_job_log(job_to_kill)

            # Removing job from job list
            if type(job_to_kill['GPU']) == list:
                for GPU in job_to_kill['GPU']:
                    if GPU in self.job_gpus:
                        self.job_gpus.remove(GPU)
            else:
                if job_to_kill['GPU'] in self.job_gpus:
                    self.job_gpus.remove(job_to_kill['GPU'])
            self.update_available_gpus()

            ElasticLogger().write_log('INFO', "Job killed", {'experiment_name': job_to_kill['experiment_name'],
                                                           'log_snapshot': job_to_kill['log_snapshot']}, push_bulk=True, print_log=False)
            self.running_jobs.remove(job_to_kill)
            self.log_handles.pop(job_to_kill['log_file'])

        elif config['operation'] == 'resources to spare':
            self.resources_to_spare = config['resources_to_spare']
        elif config['operation'] == 'restart runner':
            logger.debug('starting restart!!')
            self.close_all_logs()
            self.close_existing_connection()

            with open('runner_state_' + self.channel_tags[0] + '.pkl', 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            channels = ' '.join([' --channel ' + channel for channel in self.channel_tags])
            bash_command = 'python scripts/multiqa/allennlp_job_runner.py ' + self.resource_type + channels + \
                           ' --models_dir ' + self._MODELS_DIR + ' --state runner_state_' + self.channel_tags[0] + '.pkl'
            if self._DEBUG:
                bash_command += ' --debug '

            bash_command = 'nohup ' + bash_command + ' > logs/runner_' + self.channel_tags[0] + '.log &'
            logger.debug('running new job runners: %s',bash_command)
            proc_info = Popen(bash_command, shell=True)
            time.sleep(1)
            exit(0)

    def sample_queues(self):
        ### Reading one job from queue (by order of channel specificity)
        for channel in self.channel_tags:
            if not self.resources_available and channel == 'GPUs':
                continue

            if len(self.available_gpus)<4 and channel == '4GPUs':
                continue

            method_frame, properties, body_ = self.channel.basic_get(channel)
            if body_ is not None:
                # Display the message parts
                name = properties.headers['name']
                config = json.loads(body_.decode())
                if config['operation'] == 'kill job':
                    job_to_kill = [job for job in self.running_jobs if job['experiment_name'] == config['experiment_name']]
                    if len(job_to_kill) == 0:
                        body_ = None
                        self.channel.basic_nack(method_frame.delivery_tag)
                    else:
                        break
                elif config['operation'] != 'run job':  # no resources needed jobs
                    # we always ack for no resource jobs...
                    break
                elif name in self.runned_job_names and config['retry'] > 0:
                    # checking if this job has already been run, if so
                    # let another host try.
                    body_ = None
                    self.channel.basic_nack(method_frame.delivery_tag)
                    time.sleep(10)
                elif self.resources_available and ((not hasattr(self, '_job_no_run_string')) or self._job_no_run_string == '' or name.find(self._job_no_run_string) == -1) and  \
                        not (config['bash_command'].find('[GPU_ID4]') > -1 and len(self.available_gpus)<4):
                    break
                else:
                    # NO Resources
                    body_ = None
                    self.channel.basic_nack(method_frame.delivery_tag)
                    if not self._DEBUG:
                        time.sleep(10)

        if body_ is not None:
            logger.info('New job found! %s',name)
            logger.info('New job tag %s', method_frame.delivery_tag)

            self.channel.basic_ack(method_frame.delivery_tag)

            # performing git pull before each execution
            if not self._DEBUG:
                call("git pull origin master", shell=True, preexec_fn=os.setsid)
                time.sleep(2)

            self.handle_job_types(config, name, channel)

    def perform_iteration(self):
        # checking current iteration resource status
        self.resources_available = False
        if self.resource_type == 'GPU' and len(self.available_gpus) > self.resources_to_spare:
            self.resources_available = True
        elif self._SIM_GPUS and len(self.available_gpus) > self.resources_to_spare:
            self.resources_available = True
        elif not self._SIM_GPUS and self.resource_type == 'CPU':
            self.resources_available = True

        # checking status of all processes
        for job in self.running_jobs:
            job = self.inspect_job(job)

            if not job['alive'] or job['log_update_diff'] > 6000:
                self.handle_job_stopped(job)

        # sample queues for new jobs
        self.sample_queues()

        if self.runner_iter_count % 10 == 1:
            self.write_status()

    def run(self):
        ElasticLogger().write_log('INFO', "runner started", {'version': RUNNER_VERSION}, print_log=True)
        while True:
            if self._DEBUG:
                self.perform_iteration()
            else:
                try:
                    self.perform_iteration()
                except SystemExit:
                    exit(0)
                except:
                    # something went wrong
                    ElasticLogger().write_log('INFO', "job runner exception", {'error_message': traceback.format_exc()}, print_log=True)

                    if self.last_exception is not None and time.time() - self.last_exception < 30:
                        time.sleep(10)

                    time.sleep(3)
                    self.connect_to_queue()
                    self.last_exception = time.time()




            logger.info('%d job currently running, available GPUS: %s, job_gpus: %s', \
                        len(self.running_jobs), str(self.available_gpus), str(self.job_gpus))
            self.runner_iter_count+=1
            if not self._DEBUG:
                time.sleep(10)
            else:
                time.sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, default='gpu', help="gpu/cpu")
    parser.add_argument("--channel", type=str, help="RabbitMQ channel", action='append')
    parser.add_argument("--state", type=str, default=None, help="saved runner state")
    parser.add_argument("--models_dir", type=str, default='', help="debug session, runs dummy commands etc... ")
    parser.add_argument("--resources_to_spare", type=int, default=0, help="how many resources to spare in this machine ")
    parser.add_argument("--debug", type=bool, default=False, nargs='?', const=True ,help="debug session, runs dummy commands etc... ")
    parser.add_argument("--simulate_gpus", type=bool, default=False, nargs='?', const=True, help="debug session, runs dummy commands etc... ")
    parser.add_argument("--job_no_run_string", type=str, default='', help="debug session, runs dummy commands etc... ")
    args = parser.parse_args()

    if args.state is not None:
        with open(args.state,'rb') as f:
            runner = pickle.load(f)
        runner.connect_to_queue()
        runner.reopen_all_logs()
    else:
        runner = JobRunner(args.channel, args.type, args.models_dir, \
                           args.resources_to_spare, args.debug, args.simulate_gpus, args.job_no_run_string)
    runner.run()

if __name__ == '__main__':
    main()