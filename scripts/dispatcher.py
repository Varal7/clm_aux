import argparse
import subprocess
import os
import multiprocessing
import hashlib
import json5
import json
import sys
from os.path import dirname, realpath
import random
from typing import Dict
sys.path.append(dirname(dirname(realpath(__file__))))

count_failed = 0

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

EXPERIMENT_CRASH_MSG = bcolors.FAIL + "ALERT! job:[" + bcolors.ENDC + "{}" + bcolors.FAIL + " ] has crashed! Check logfile at:\n{}\nComet: {}" + bcolors.ENDC
CONFIG_NOT_FOUND_MSG = "ALERT! {} config {} file does not exist!"
RESULTS_PATH_APPEAR_ERR = 'results_path should not appear in config. It will be determined automatically per job'
SUCESSFUL_SEARCH_STR = bcolors.OKCYAN + "SUCCESS! Check logfile at:\n{}\nComet: {}" + bcolors.ENDC
LAUNCH_EXP_STR = bcolors.OKGREEN + "Launched exp: \t" + bcolors.OKBLUE + "({})"+ bcolors.ENDC + "\n" + "{}"
SKIPPED_EXP_STR = bcolors.WARNING + "Skipped exp because [{}] \n" + "Log file: {}\nResults path: {}"+ bcolors.ENDC + "\n"
POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'

parser = argparse.ArgumentParser(description='Sandstone Grid Search Dispatcher. For use information, see `doc/README.md`')
parser.add_argument("experiment_config_path", type=str, help="Path of experiment config")
parser.add_argument('--script', type=str, default="scripts/gnn.py", help="Script to run")
parser.add_argument('--log_dir', type=str, default="/Mounts/rbg-storage1/logs/repg2", help="path to store logs and detailed job level result files")
parser.add_argument('--rerun_experiments', action='store_true', default=False, help='whether to rerun experiments with the same result file location')
parser.add_argument('--shuffle_experiment_order', action='store_true', default=False, help='whether to shuffle order of experiments')
parser.add_argument('--dry_run', action='store_true', default=False, help='whether to not actually run the jobs')
parser.add_argument('--prefix', type=str, default="", help='Prefix for results path')
parser.add_argument('--limit_jobs', type=int, default=None, help='Limit number of jobs to run')


def md5(key):
    return hashlib.md5(key.encode()).hexdigest()

def parse_search_space(search_space, jobs):
    """
    Parse a search space and expands jobs, which is a list of flag sstrings
    Typically, you want to initialize jobs = [("", "")].
    """
    for flag, possible_values in search_space.items():
        if len(possible_values) == 0 or type(possible_values) is not list:
            raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
        children = []
        if flag == "dispatcher_prefix":
            children = [(possible_values[0], job) for (_, job) in jobs]
        else:
            for value in possible_values:
                for old_prefix, parent_job in jobs:
                    if type(value) is bool:
                        if value:
                            new_job_str = "{} --{}".format(parent_job, flag)
                        else:
                            new_job_str = parent_job
                    elif type(value) is list:
                        val_list_str = " ".join([str(v) for v in value])
                        new_job_str = "{} --{} {}".format(parent_job, flag,
                                                            val_list_str)
                    else:
                        new_job_str = "{} --{} {}".format(parent_job, flag, value)
                    children.append((old_prefix, new_job_str))
        jobs = children

    return jobs

def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of (prefix, flag strings), each of which encapsulates one job.
        *Example: ("with-gpu", "--train --cuda --dropout=0.1 ...")
    '''
    main_search_space = config['main_search_space']
    sub_search_spaces = config['sub_search_spaces'] if 'sub_search_spaces' in config else []

    main_jobs = parse_search_space(main_search_space, [("", "")])

    if len(sub_search_spaces) == 0:
        return main_jobs

    all_jobs = []
    for search_space in sub_search_spaces:
        all_jobs.extend(parse_search_space(search_space, main_jobs))

    return all_jobs


def launch_experiment(args, gpu, worker_args, flag_string, job_prefix):
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    Alert of something goes wrong.
    :gpu: gpu to run this machine on.
    :flag_string: flags to use for this model run. Will be fed into
    :prefix: prefix to use for result path
    scripts/main.py
    '''
    global count_failed
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_name = args.prefix + job_prefix + md5(flag_string)
    log_stem = os.path.join(args.log_dir, log_name)
    log_path = '{}.txt'.format(log_stem)
    results_path = "{}.results".format(log_stem)

    experiment_string = f"CUDA_VISIBLE_DEVICES={gpu} python -u {args.script} {flag_string} --results_path {results_path}"

    if 'port' in worker_args:
        experiment_string += ' --master_port {}'.format(worker_args['port'])

    if 'host' in worker_args:
        experiment_string += ' --master_host {}'.format(worker_args['host'])


    # forward logs to logfile
    if "--resume" in flag_string and not args.rerun_experiments:
        pipe_str = ">>"
    else:
        pipe_str = ">"

    shell_cmd = "{} {} {} 2>&1".format(experiment_string, pipe_str, log_path)
    print(LAUNCH_EXP_STR.format(log_path, shell_cmd))

    if args.dry_run:
        print(SKIPPED_EXP_STR.format("--dry_run", log_path, results_path))

    elif os.path.exists(results_path) and not args.rerun_experiments:
        print(SKIPPED_EXP_STR.format("results_path exists", log_path, results_path))

    else:
        subprocess.call(shell_cmd, shell=True)

        if not check_done(results_path):
            # running this process failed
            job_fail_msg = EXPERIMENT_CRASH_MSG.format(experiment_string, log_path, get_comet_url(results_path))
            count_failed += 1
            print(job_fail_msg)

        else:
            print(SUCESSFUL_SEARCH_STR.format(log_path, get_comet_url(results_path)))

    return results_path, log_path

def get_comet_url(results_path):
    if not os.path.exists(results_path):
        return None

    result_dict = json.load(open(results_path))
    return result_dict['comet_url'] if 'comet_url' in result_dict else None

def check_done(results_path):
    if not os.path.exists(results_path):
        return False

    result_dict = json.load(open(results_path, 'rb'))
    return result_dict['status'] == "done"

def worker(args, gpu, worker_args, job_queue, done_queue):
    '''
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        prefix, params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, gpu, worker_args, params, prefix))

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.experiment_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.experiment_config_path))
        sys.exit(1)
    experiment_config: Dict = json5.load(open(args.experiment_config_path, 'r'))

    if 'results_path' in experiment_config['main_search_space']:
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)

    job_list = parse_dispatcher_config(experiment_config)
    if args.shuffle_experiment_order:
        random.shuffle(job_list)
    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    if args.limit_jobs is not None:
        job_list = job_list[:args.limit_jobs]

    for job in job_list:
        job_queue.put(job)
    print("Launching Dispatcher with {} jobs!".format(len(job_list)))
    print()
    for worker_indx, gpu in enumerate(experiment_config['available_gpus']):
        print("Start gpu worker {}".format(gpu))
        worker_args = {}
        if 'ports' in experiment_config:
            worker_args['port'] = experiment_config['ports'][worker_indx]
        if 'hosts' in experiment_config:
            worker_args['host'] = experiment_config['hosts'][worker_indx]
        multiprocessing.Process(target=worker, args=(args, gpu, worker_args, job_queue, done_queue)).start()
    print()

    summary = []

    for i in range(len(job_list)):
        result_path, log_path = done_queue.get()
        print("({}/{}) \t {}".format(i+1, len(job_list), "Done!"))

    # return non-zero error code if one of the jobs failed
    assert count_failed == 0
