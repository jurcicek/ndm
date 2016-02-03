import json
import os
from datetime import datetime

exp_dir = ''


def dt():
    return datetime.now().isoformat().replace('T', '--').replace(':', '-')


def prepare_experiment(FLAGS):
    global exp_dir

    experiment_name = '{date_time}' \
                      '--model={model}' \
                      '--data_fraction={data_fraction}' \
                      '--max_epochs={max_epochs}' \
                      '--batch_size={batch_size}' \
                      '--learning_rate={learning_rate:e}' \
                      '--dropout_keep_prob={dropout_keep_prob:e}'.format(
        date_time=dt(),
        model=FLAGS.model,
        data_fraction=FLAGS.data_fraction,
        max_epochs=FLAGS.max_epochs,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        dropout_keep_prob=FLAGS.dropout_keep_prob,
    )

    exp_dir = os.path.join('./experiments', experiment_name)

    try:
        os.mkdir(exp_dir)
    except FileExistsError:
        pass

    os.system("mkdir {d}/ndm".format(d=exp_dir))
    os.system("cp ./*.py {d}/ndm".format(d=exp_dir))
    os.system("cp ./*.json {d}/ndm".format(d=exp_dir))
    os.system("mkdir {d}/tfx".format(d=exp_dir))
    os.system("cp ../tfx/*.py {d}/tfx".format(d=exp_dir))

    return exp_dir


def start_experiment(run):
    global exp_dir

    exp_dir = os.path.join(exp_dir, 'run-{d:03}'.format(d=run))
    try:
        os.mkdir(exp_dir)
    except FileExistsError:
        pass
    print('Log directory: {exp_dir}'.format(exp_dir=exp_dir))


class LogMessage:
    filename = 'log.txt'

    def __init__(self, msg=None, time=False, log_fn='log.txt'):
        if msg:
            self.msg = [msg, ]
        else:
            self.msg = []
        self.time = time
        self.filename = log_fn

    def add(self, m=''):
        self.msg.append(str(m))

    @staticmethod
    def write(msg):
        msg = str(msg)
        with open(os.path.join(exp_dir, LogMessage.filename), 'ta') as l:
            l.write(msg)

        print(msg, end='', flush=True)

    def log(self, end='\n', print_console=True):
        msg = end.join(self.msg)
        with open(os.path.join(exp_dir, self.filename), 'ta') as l:
            if self.time:
                l.write('Time stamp: {s}'.format(s=dt()))
                l.write('\n')

            l.write(msg)
            l.write('\n')

        if print_console:
            print(msg)

        self.msg = []


class LogExperiment:
    filename = 'experiment.json'

    def __init__(self, results):
        with open(os.path.join(exp_dir, LogExperiment.filename), 'tw') as js:
            json.dump(results, js, sort_keys=True, indent=4, separators=(',', ': '))

def read_experiment(run):
    with open(os.path.join(exp_dir, 'run-{d:03}'.format(d=run), 'experiment.json'), 'r') as js:
        return json.load(js)

