
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
parser.add_argument("-s", "--shell_type", type=str, default='bash',
                        help="shell_type")
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

            # Log snapshot
            with open(proc['log_file'], 'r') as f:
                log_data = f.readlines()
                proc['log_snapshot'] = ' '.join(log_data[-100:])

            # Log time out handling TODO
            statbuf = os.stat(proc['log_file'])
            proc['log_update_diff'] = time.time() - statbuf.st_mtime
            #proc['log_update_diff'] > 2000:

            if not proc['alive']:
                # checking if process successfully completed task,
                # TODO this is an ugly check, but because we are forking with nohup, python does not provide any good alternative...
                if proc['log_snapshot'].find('Traceback (most recent call last):') > -1:
                    ElasticLogger().write_log('INFO', "Job died", proc, push_bulk=True,print_log=True)

                    # Requeue
                    channel.basic_nack(proc['job_tag'])
                    proc_running.remove(proc)
                    break
                else:
                    ElasticLogger().write_log('INFO', "Job finished successfully", proc, push_bulk=True,print_log=True)

                    # ack
                    channel.basic_ack(proc['job_tag'])
                    proc_running.remove(proc)
                    break


        ### Reading one job from queue
        method_frame, properties, body = channel.basic_get(args.channel)

        if body is not None:

            # Display the message parts
            print(method_frame)
            print(properties)
            print(body)


            log_file = properties.headers['name'] + '.txt'
            if args.shell_type == 'bash':
                command = 'nohup ' + body.decode() + ' > ' + log_file + ' &'
            else:
                command = 'nohup ' + body.decode() + ' >& ' + log_file

            wa_proc = Popen(command, shell=True, preexec_fn=os.setsid)
            proc_running.append({'job_tag':method_frame.delivery_tag,'command':command, \
                                 'log_file':log_file, \
                                 'name':properties.headers['name'],'alive': True,\
                                 'pid': wa_proc.pid+1, 'start_time': time.time()})

            time.sleep(2)

        if iter_count % 10 == 1:
            # Virtual memory usage
            # Giving info on the processes running and GPU status

            # resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            for proc in proc_running:
                try:
                    proc['memory'] = psutil.Process(proc['pid']).memory_info()
                    proc['command'] = ' '.join(psutil.Process(wa_proc.pid+1).cmdline())
                except:
                    proc['alive'] = False

            ElasticLogger().write_log('INFO', "GPU machine status", {'gpus':gpu_memory_mb(),'procs_running': proc_running,\
                                                      'num_procs_running':len(proc_running),}, push_bulk=True,print_log=True)

        time.sleep(2)


    except:
        channel.close()
        connection.close()

        # something went wrong
        time.sleep(3)
        print(traceback.format_exc())
        ElasticLogger().write_log('INFO', "job runner exception", {'error_message': traceback.format_exc()}, push_bulk=True,print_log=True)
        print('no internet connection? ')
