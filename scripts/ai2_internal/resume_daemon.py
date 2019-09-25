#! /usr/bin/env python

# Tool to automatically resume preemptible beaker experiments.

from sqlite3 import Connection
from enum import Enum
import argparse
import json
import logging
import os
import sqlite3
import subprocess
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(
        f"{os.environ['HOME']}/.allennlp/resume.log",
        maxBytes=1024*1024,
        backupCount=10)
logger.addHandler(handler)

BEAKER_QUERY_INTERVAL_SECONDS = 1

# See https://github.com/beaker/client/blob/master/api/task_status.go
class BeakerStatus(Enum):
    submitted = "submitted"
    provisioning = "provisioning"
    initializing = "initializing"
    running = "running"
    terminating = "terminating"
    preempted = "preempted"
    succeeded = "succeeded"
    skipped = "skipped"
    stopped = "stopped"
    failed = "failed"

    def __str__(self):
        return self.name

    def is_end_state(self):
        if self is BeakerStatus.preempted:
            return True
        elif self is BeakerStatus.succeeded:
            return True
        elif self is BeakerStatus.skipped:
            return True
        elif self is BeakerStatus.stopped:
            return True
        elif self is BeakerStatus.failed:
            return True
        else:
            return False

class BeakerWrapper:

    def get_status(self, experiment_id: str) -> BeakerStatus:
        #brendanr.local ➜  ~ beaker experiment inspect ex_g7knlblsjxxk
        #[
        #    {
        #        "id": "ex_g7knlblsjxxk",
        #        "owner": {
        #            "id": "us_a4hw8yvr3xut",
        #            "name": "ai2",
        #            "displayName": "AI2"
        #        },
        #        "author": {
        #            "id": "us_hl8x796649u9",
        #            "name": "brendanr",
        #            "displayName": "Brendan Roof"
        #        },
        #        "workspace": "",
        #        "user": {
        #            "id": "",
        #            "name": "",
        #            "displayName": ""
        #        },
        #        "nodes": [
        #            {
        #                "name": "training",
        #                "task_id": "",
        #                "taskId": "tk_64wm85lc3f0m",
        #                "result_id": "",
        #                "resultId": "ds_du02un92r57b",
        #                "status": "initializing",
        #                "child_task_ids": null,
        #                "childTaskIds": [],
        #                "parent_task_ids": null,
        #                "parentTaskIds": []
        #            }
        #        ],
        #        "created": "2019-09-25T02:03:30.820437Z",
        #        "archived": false
        #    }
        #]

        command = ["beaker", "experiment", "inspect", experiment_id]
        experiment_json = subprocess.check_output(command)
        experiment_data = json.loads(experiment_json)
        # TODO(brendanr): Are these constraints ever actually violated?
        assert len(experiment_data) == 1
        assert len(experiment_data[0]["nodes"]) == 1
        status = BeakerStatus(experiment_data[0]["nodes"][0]["status"])
        return status
        # sleep 1 (help avoid thrashing beaker)

    def resume(self, experiment_id: str) -> str:
        command = ["beaker", "experiment", "resume", f"--experiment-name={experiment_id}"]
        return subprocess.check_output(command, universal_newlines=True).strip()
        # sleep 1 (help avoid thrashing beaker)

def create_table(connection: Connection) -> None:
    cursor = connection.cursor()
    create_table_statement = """
    CREATE TABLE active_experiments
    (experiment_id TEXT PRIMARY KEY, original_id TEXT, max_resumes INTEGER, current_resume INTEGER)
    """
    cursor.execute(create_table_statement)
    connection.commit()


def start_autoresume(connection: Connection, experiment_id: str, max_resumes: int) -> None:
    cursor = connection.cursor()
    cursor.execute(
            "INSERT INTO active_experiments VALUES (?, ?, ?, ?)",
            (experiment_id, experiment_id, max_resumes, 0))
    connection.commit()


def stop_autoresume(connection: Connection, experiment_id: str) -> None:
    cursor = connection.cursor()
    cursor.execute(
            "SELECT * FROM active_experiments WHERE experiment_id = ?",
            (experiment_id,))
    result = cursor.fetchall()
    assert result, f'Experiment {experiment_id} not found!'
    cursor.execute(
            'DELETE FROM active_experiments WHERE experiment_id = ?',
            (experiment_id,))
    connection.commit()


def resume(connection: Connection, beaker: BeakerWrapper) -> None:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM active_experiments")
    experiments = cursor.fetchall()

    for experiment_row in experiments:
        experiment_id, original_id, max_resumes, current_resume = experiment_row
        status = beaker.get_status(experiment_id)
        if status.is_end_state():
            stop_autoresume(connection, experiment_id)
            if status is BeakerStatus.preempted:
                if current_resume >= max_resumes:
                    logger.info(f"Experiment {experiment_id} preempted too many times "
                            f"({max_resumes}). Original experiment: {original_id}")
                else:
                    new_experiment_id = beaker.resume(experiment_id)
                    logger.info(f"Experiment {experiment_id} preempted "
                            f"({current_resume}/{max_resumes}). Resuming as: "
                            f"{new_experiment_id} Original experiment: {original_id}")
                    cursor.execute(
                            "INSERT INTO active_experiments VALUES (?, ?, ?, ?)",
                            (new_experiment_id, original_id, max_resumes, current_resume + 1))
                    connection.commit()
            else:
                logger.info(f"Experiment {experiment_id} completed with status: "
                        f"{status}. Original experiment: {original_id}")


class Action(Enum):
    install = "install"
    start = "start"
    stop = "stop"
    resume = "resume"

    def __str__(self):
        return self.name


def main(args) -> None:
    db_path = os.environ["HOME"] + "/.allennlp/resume.db"
    connection = sqlite3.connect(db_path)

    # TODO(brendanr): Just do this automatically?
    if args.action is Action.install:
        create_table(connection)
        # TODO(brendanr): Put resume_daemon.py --action=resume in the users
        # crontab. You'll need to copy the exist one with `crontab -l` or
        # similar into a temp file, then append that line and reinstall with
        # `crontab file` Also grep for resume_daemon.py in the crontab.
        return

    if args.action is Action.start:
        start_autoresume(connection, args.experiment_id, args.max_resumes)
    elif args.action is Action.stop:
        stop_autoresume(connection, args.experiment_id)
    elif args.action is Action.resume:
        beaker = BeakerWrapper()
        resume(connection, beaker)
    else:
        raise Exception(f"Unaccounted for action {action}")
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=Action, choices=list(Action), required=True)
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--max-resumes", type=int, default=10)
    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        # Ensure traces are logged.
        # TODO(brendanr): Is there a better way to do this?
        logger.exception("Fatal error")
        raise
