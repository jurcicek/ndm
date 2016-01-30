import os
from datetime import datetime

exp_dir = ''


def dt():
    return datetime.now().isoformat().replace('T', '--').replace(':', '-')


def start_experiment(FLAGS):
    global exp_dir

    date_time = dt()
    exp_dir = './experiments/{date_time}' \
              '--model={model}' \
              '--data_fraction={data_fraction}' \
              '--max_epochs={max_epochs}' \
              '--batch_size={batch_size}' \
              '--learning_rate={learning_rate}' \
              '--dropout_keep_prob={dropout_keep_prob}'.format(
        date_time=date_time,
        model=FLAGS.model,
        data_fraction=FLAGS.data_fraction,
        max_epochs=FLAGS.max_epochs,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        dropout_keep_prob=FLAGS.dropout_keep_prob,
    )
    print('Log directory: {exp_dir}'.format(exp_dir=exp_dir))

    os.mkdir(exp_dir)
    os.system("mkdir {d}/ndm".format(d=exp_dir))
    os.system("cp ./*.py {d}/ndm".format(d=exp_dir))
    os.system("cp ./*.json {d}/ndm".format(d=exp_dir))
    os.system("mkdir {d}/tfx".format(d=exp_dir))
    os.system("cp ../tfx/*.py {d}/tfx".format(d=exp_dir))


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

