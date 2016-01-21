#!/usr/bin/env python3
import glob
import json
import os

import wget


def download_dstc2_data():
    if not os.path.isfile('./tmp/dstc2_traindev.tar.gz'):
        wget.download('http://camdial.org/~mh521/dstc/downloads/dstc2_traindev.tar.gz', './tmp')
        wget.download('http://camdial.org/~mh521/dstc/downloads/dstc2_test.tar.gz', './tmp')
        wget.download('http://camdial.org/~mh521/dstc/downloads/dstc2_scripts.tar.gz', './tmp')
        print()
    else:
        print('Everything appears to be downloaded.')


def unpack():
    os.makedirs('./tmp/dstc2_traindev', exist_ok=True)
    os.makedirs('./tmp/dstc2_test', exist_ok=True)
    if not os.path.isdir('./tmp/dstc2_traindev/data'):
        os.system('tar xvzf ./tmp/dstc2_traindev.tar.gz -C ./tmp/dstc2_traindev')
        os.system('tar xvzf ./tmp/dstc2_test.tar.gz -C ./tmp/dstc2_test')
        os.system('tar xvzf ./tmp/dstc2_scripts.tar.gz -C ./tmp')


def gen_slot(slot, goal):
    # if slot in goal:
    #     return slot + '-' + goal[slot].replace(' ', '-')
    # else:
    #     return slot + '-' + 'none'
    if slot in goal:
        return goal[slot]
    else:
        return 'none'


def extract_data(file_name):
    conversation = []
    with open(file_name) as flabel:
        with open(file_name.replace('label.json', 'log.json')) as flog:
            label = json.load(flabel)
            log = json.load(flog)

            for lab_turn, log_turn in zip(label['turns'], log['turns']):
                system = log_turn['output']['transcript']
                user = lab_turn['transcription']
                user_asr = log_turn['input']['live']['asr-hyps'][0]['asr-hyp']
                user_asr_score = log_turn['input']['live']['asr-hyps'][0]['score']

                state = []
                state.append(gen_slot('food', lab_turn['goal-labels']))
                state.append(gen_slot('area', lab_turn['goal-labels']))
                state.append(gen_slot('pricerange', lab_turn['goal-labels']))
                # state.append(gen_slot('signature', lab_turn['goal-labels']))
                # state.append(gen_slot('postcode', lab_turn['goal-labels']))
                # state.append(gen_slot('addr', lab_turn['goal-labels']))
                # state.append(gen_slot('phone', lab_turn['goal-labels']))
                # state.append(gen_slot('name', lab_turn['goal-labels']))
                state = ' '.join(state)

                print(system)
                print(user)
                print(user_asr)
                print(user_asr_score)
                print(state)
                print('-' * 120)

                conversation.append((system, user, user_asr, user_asr_score, state))
    return conversation


def gen_data(dir_name):
    conversations = []
    jsons = []
    jsons.extend(glob.glob(os.path.join(dir_name, '*/label.json')))
    jsons.extend(glob.glob(os.path.join(dir_name, '*/*/label.json')))
    jsons.extend(glob.glob(os.path.join(dir_name, '*/*/*/label.json')))

    for js in jsons:
        print('Processing a file: {fn}'.format(fn=js))
        conversations.append(extract_data(js))

    return conversations


if __name__ == '__main__':
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    download_dstc2_data()
    unpack()

    conversations = gen_data('./tmp/dstc2_test/data')
    with open('./data.dstc2.test.json', 'w') as f:
        json.dump(conversations, f, sort_keys=True, indent=4, separators=(',', ': '))

    conversations = gen_data('./tmp/dstc2_traindev/data')
    with open('./data.dstc2.traindev.json', 'w') as f:
        json.dump(conversations, f, sort_keys=True, indent=4, separators=(',', ': '))
