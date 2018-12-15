#from webkb_config import *
from answer_batch import AnswerBatch
from webkb_config import *
from subprocess import PIPE, Popen
import sys, traceback
from threading  import Thread
import time
from Queue import Queue, Empty
import dropbox
import socket
import shutil
import argparse
import signal

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("MAX_NUM_OF_PROC")
parser.add_argument("delay_rounds")
args = parser.parse_args()
args.MAX_NUM_OF_PROC = int(args.MAX_NUM_OF_PROC)
args.delay_rounds = int(args.delay_rounds)

def enqueue_output(out,err,q):
    for line in iter(out.readline, b''):
        q.put(line)
    for line in iter(err.readline, b''):
        q.put('error: ' + line)
    q.put('exit')
    out.close()

dirs_to_process = []
proc_running = []
dbx = dropbox.Dropbox('7j6m2s1jYC0AAAAAAAHy69fu0OxDAU3fPbIjjarqr_1zalj8Mvypf8U71BoLT-AD')

# searching inside the webanswer batch output
iter_count = 0
found_counter = 0
while True:
    try:
        iter_count+=1
        for dirname, dirnames, filenames in os.walk('batch_output'):
            # checking status of all processes
            for proc in proc_running:
                if not os.path.isfile("batch_output/" + proc['dirname'] + "/processing.txt"):
                    print ('proc finished - removing it')
                    proc_running.remove(proc)
                    break
                else:
                    try:
                        os.killpg(os.getpgid(proc['pid']), 0)
                    except:
                        proc['alive'] = False
                    # checking log last update time
                    statbuf = os.stat("batch_output/" + proc['dirname'] + "/log.txt")
                    update_diff = time.time() - statbuf.st_mtime
                    if update_diff>1000 or not proc['alive'] or time.time()-proc['start_time']>3600:
                        elastic.log('ERROR', "proc likely dead - restarting", {'proc':proc['dirname'], 'computer': socket.gethostname()}, push_bulk=True)
                        if proc['alive']:
                            os.killpg(os.getpgid(proc['pid']), signal.SIGTERM)

                        if proc['retry']<3:
                            log_file = 'batch_output/' + proc['dirname'] + '/log.txt'
                            command = "nohup python webanswer_single_batch.py answer_batch webCompQ train --full_batchname " + proc['dirname'] + " >& " + log_file
                            wa_proc = Popen(command, shell=True, preexec_fn=os.setsid)
                            time.sleep(3)
                            proc['pid'] = wa_proc.pid
                            proc['retry'] += 1
                            proc['alive'] = True
                        else:
                            elastic.log('ERROR', "more than 3 proc retries!",
                                        {'proc': proc['dirname'], 'computer': socket.gethostname()}, push_bulk=True)
                            try:
                                shutil.move(log_file, 'error_logs/' + proc['dirname'] + '_error_log.txt')
                                shutil.rmtree('batch_output/' + proc['dirname'])
                            except:
                                print('could not rmtree!!')
                            proc_running.remove(proc)

                        #os.remove("batch_output/" + proc['dirname'] + "/processing.txt")

                        break
			

            if len(proc_running)>=args.MAX_NUM_OF_PROC:
                continue

            if 'googled.json' not in filenames and 'features.json' not in filenames:
                continue

            if 'processing.txt' in filenames:
                continue

            dirs_to_process = dirname.replace('batch_output/','')
            print ('found new process to run in ' + dirs_to_process)

            with open('batch_output/' + dirs_to_process + '/processing.txt', 'w') as f:
                json.dump([], f)

            log_file = 'batch_output/' + dirs_to_process + '/log.txt'
            print ('starting process')
            command = "nohup python webanswer_single_batch.py answer_batch webCompQ train --full_batchname " + dirs_to_process + " >& " + log_file
            print(command)
            wa_proc = Popen(command, shell=True, preexec_fn=os.setsid)
            proc_running.append({'dirname': dirs_to_process,'alive':True,'pid':wa_proc.pid,'retry':0,'start_time':time.time()})
            time.sleep(3)

        file_to_word_on = None
        for entry in dbx.files_list_folder('/google').entries:
            if len(proc_running) < args.MAX_NUM_OF_PROC and entry.name.find('_done.json') > -1:
                found_counter += 1
                if found_counter>args.delay_rounds:
                    found_counter=0
                    batch_dir = 'batch_output/' + entry.name.replace('_done.json', '') + '/'
                    cache_dir_available = os.path.exists(batch_dir)
                    if not cache_dir_available:
                            os.makedirs(batch_dir)

                    # copying file to backup and backupdir
                    md, res = dbx.files_download('/google/' + entry.name)
                    with open(batch_dir + 'googled.json', 'w') as f:
                            f.write(res.content)

                    # moving the file to backup
                    dbx.files_move('/google/' + entry.name, '/cache/' \
                            + datetime.datetime.fromtimestamp(time.time()).strftime(
                            '%Y-%m-%d_%H_%M_%S') + '__' + entry.name)
                    break

        if iter_count % 10 == 1:
            elastic.log('INFO', "multi batch status", {'procs_running': proc_running, 'computer': socket.gethostname(),\
                                                      'num_procs_running':len(proc_running)}, push_bulk=True)

        # specific to university computers:
        if iter_count % 1000 == 1:
            try:
                shutil.rmtree(
                    '/specific/disk1/home/alont/theano/compiledir_Linux-4.9--net1-x86_64-with-Ubuntu-14.04-trusty-x86_64-2.7.6-64')
            except:
                print('failed removing theano compile dir')

        time.sleep(10)


    except:
        time.sleep(3)
        print(traceback.format_exc())
        elastic.log('ERROR', "multi batch exception", {'error_message': traceback.format_exc()}, push_bulk=True)
        print('no internet connection? ')
