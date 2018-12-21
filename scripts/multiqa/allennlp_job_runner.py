
## Version 0.0
#  only takes a command string and runs it on a specific channel, channel is the GPU computer name...
from allennlp.common.util import *
from subprocess import Popen
import traceback,os, psutil
import time
import socket
import shutil
import argparse
import signal
from allennlp.common.elastic_logger import ElasticLogger
import pika
connection_params = pika.URLParameters('amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
connection = pika.BlockingConnection(connection_params)
channel = connection.channel()

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("channel", type=str,
                        help="RabbitMQ channel")
parser.add_argument("--shell", type=str, default="not_bash",
                        help="RabbitMQ channel")
args = parser.parse_args()

proc_running = []
iter_count = 0 # counting iteration for writing status
while True:
    try:
        iter_count+=1

        # checking status of all processes
        for proc in proc_running:
            # check if process still alive:
            try:
                print('checcking if process alive '  + str(proc['pid']))
                os.killpg(os.getpgid(proc['pid']), 0)
            except:
                proc['alive'] = False

            try:
                # Log snapshot
                with open(proc['log_file'], 'r') as f:
                    log_data = f.readlines()
                    proc['log_snapshot'] = ' '.join(log_data[-50:])
            except:
                proc['alive'] = False


            # Log time out handling TODO
            try:
                statbuf = os.stat(proc['log_file'])
                proc['log_update_diff'] = time.time() - statbuf.st_mtime
            except:
                print('file not found')
            #proc['log_update_diff'] > 2000:

            if not proc['alive']:
                # checking if process successfully completed task,
                # TODO this is an ugly check, but because we are forking with nohup, python does not provide any good alternative...
                if proc['log_snapshot'].find('Traceback (most recent call last):') > -1 or \
                    proc['log_snapshot'].find('error') > -1:
                    ElasticLogger().write_log('INFO', "Job died", proc, push_bulk=True,print_log=True)

                    # Requeue
                    #channel.basic_nack(proc['job_tag'])
                    proc_running.remove(proc)
                    break
                else:
                    ElasticLogger().write_log('INFO', "Job finished successfully", proc, push_bulk=True,print_log=True)

                    # ack
                    #channel.basic_ack(proc['job_tag'])
                    proc_running.remove(proc)
                    break


        ### Reading one job from queue
        method_frame, properties, body = channel.basic_get(args.channel)

        if body is not None:

            # Display the message parts
            body = body.decode()
            print(method_frame)
            print(properties)
            print(body)

            #if properties.headers['name'] == 'restart':
                #os.execl(path, arg0, arg1, ...)
            # supporting process kill command:
            if properties.headers['name'] == 'kill job':
                print('kill job!')
                try:
                    pid_to_kill = [proc['pid'] for proc in proc_running if proc['experiment_name'] == body]
                    body = 'kill ' + str(pid_to_kill[0])
                except:
                    ElasticLogger().write_log('INFO', "job runner exception", {'error_message': traceback.format_exc()},
                                              push_bulk=True, print_log=True)
                    channel.basic_ack(method_frame.delivery_tag)
                    time.sleep(2)


            log_file = 'logs/' + properties.headers['name'] + '.txt'
            if args.shell == 'bash':
                command = 'nohup ' + body + ' &'
            else:
                command = 'nohup ' + body + ' &'
            print(command)
            log_dir_part = log_file.split('/')
            for dir_depth in range(len(log_dir_part)-1):
                if not os.path.isdir('/'.join(log_dir_part[0:dir_depth+1])):
                    os.mkdir('/'.join(log_dir_part[0:dir_depth+1]))

            with open(log_file,'wb') as f:
                wa_proc = Popen(command, shell=True, preexec_fn=os.setsid,stdout=f,stderr=f)
            proc_running.append({'job_tag':method_frame.delivery_tag,'command':command, \
                                 'log_file':log_file, 'log_snapshot':'',\
                                 'experiment_name':properties.headers['name'], 'alive': True,\
                                 'pid': wa_proc.pid+1, 'start_time': time.time()})
            # we are not persistant for now ...
            channel.basic_ack(method_frame.delivery_tag)
            time.sleep(2)



        if iter_count % 10 == 1:
            # Virtual memory usage
            # Giving info on the processes running and GPU status

            # resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            for proc in proc_running:
                try:
                    proc['memory'] = psutil.Process(proc['pid']).memory_info()
                    #proc['command'] = ' '.join(psutil.Process(wa_proc.pid+1).cmdline())
                except:
                    proc['alive'] = False

                ElasticLogger().write_log('INFO', "Job Status",proc,push_bulk=True, print_log=True)

            gpu_mem = gpu_memory_mb()
            # Ugly patch for misconfigured GPUs...
            if args.channel == 'rack-jonathan-g02':
                gpu_mem = {(3 - key):val for key,val in gpu_mem.items()}
            free_gpus = [i for i, gpu in enumerate(gpu_mem.keys()) if gpu_mem[gpu] < 1700]

            ElasticLogger().write_log('INFO', "Machine Status", {'gpus':gpu_mem,'free_gpus':free_gpus,\
                                                      'num_procs_running':len(proc_running),}, push_bulk=True,print_log=True)

        time.sleep(6)


    except:
        # something went wrong
        time.sleep(3)
        print(traceback.format_exc())
        ElasticLogger().write_log('INFO', "job runner exception", {'error_message': traceback.format_exc()},
                                  push_bulk=True, print_log=True)

        # maybe this is a connection error:
        try:
            channel.close()
            connection.close()
        except:
            print('close failed')

        try:
            connection_params = pika.URLParameters(
                'amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
            print('reconnected')
        except:
            print('reconnecting failed')



channel.close()
connection.close()
