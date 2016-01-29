import os
from datetime import datetime

exp_dir = ''


def start_experiment(FLAGS):
    global exp_dir

    date_time = datetime.now().isoformat().replace('T', '--').replace(':', '-')
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
    def __init__(self):
        self.msg = []

    def add(self, m):
        self.msg.append(m)

    def log(self):
        msg = '\n'.join(self.msg)
        with open(os.path.join(exp_dir, 'log.txt'), 'ta') as l:
            l.write(msg)
            l.write('\n')

        print(msg)
