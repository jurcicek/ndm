#!/usr/bin/env python3
import json
from collections import defaultdict
from copy import deepcopy
from random import shuffle

import numpy as np
import sys

import ontology


def load_json_data(file_name):
    """Load a json file - file_name.

    :param file_name: a name of the json file
    :return: the Python representation of the json file
    """
    with open(file_name) as f:
        text_data = json.load(f)

    return text_data


def gen_examples(text_data, input):
    """Generates training examples for the conversation to sequence model. Here, we experiment with conversational models
    that is converting a conversational history into dialogue state representation (dialogue state tracking) and generation
    a textual response given the conversation history (dialogue policy).

    :param text_data: a list of conversation each composed of (system_output, user_input, dialogue state) tuples
    :return: a transformed text_data
    """
    examples = []
    for conversation in text_data:
        history = []
        scores = []
        prev_turn = None
        for turn in conversation:
            if prev_turn:
                history.append(prev_turn[0])

                if input == 'trs':
                    user_input = prev_turn[1]
                    score = 1.0
                elif input == 'asr':
                    user_input = prev_turn[2]
                    score = float(prev_turn[3])
                else:
                    raise Exception('Unsupported type of input')
                history.append(user_input)
                scores.append(score)  # the score of the last ASR user input

                state = prev_turn[4]  # the dialogue state
                action = turn[0]      # the system action / response
                examples.append(deepcopy([history, state, action]))
            prev_turn = turn

        history.append(prev_turn[0])
        history.append(prev_turn[1])
        state = prev_turn[2]  # the dialogue state
        action = 'hangup'  # the system action / response
        examples.append(deepcopy([history, state, action]))

    return examples


def get_words(utterance):
    """Splits an utterance into words, removes some characters not available in spoken dialogue systems,
    uppercases the text.

    :param utterance: a string
    :return: a list of string (words)
    """
    for c in '?!.,':
        utterance = utterance.replace(c, ' ').replace('  ', ' ')

    return utterance.lower().split()


def normalize(examples):
    norm_examples = []
    for history, state, action in examples:
        norm_history = []
        for utterance in history:
            utterance_words = get_words(utterance)
            norm_history.append(utterance_words)

        norm_state = get_words(state)
        norm_action = get_words(action)

        norm_examples.append([norm_history, norm_state, norm_action])

    return norm_examples


def sort_by_conversation_length(examples):
    examples.sort(key=lambda example: len(example[0]))

    return examples


def get_word2idx(idx2word):
    return dict([(w, i) for i, w in enumerate(idx2word)])


def count_dict(lst, dct):
    for word in lst:
        dct[word] += 1


def get_idx2word(examples):
    words_history = defaultdict(int)
    words_history_arguments = defaultdict(int)
    words_state = defaultdict(int)
    words_action = defaultdict(int)
    words_action_arguments = defaultdict(int)
    words_action_templates = defaultdict(int)

    for abs_history, history_arguments, abs_state, abs_action, action_arguments, action_templates in examples:
        for utterance in abs_history:
            count_dict(utterance, words_history)

        count_dict(history_arguments, words_history_arguments)
        count_dict(abs_state, words_state)
        count_dict(abs_action, words_action)
        count_dict(action_arguments, words_action_arguments)
        words_action_templates[action_templates] += 1

    idx2word_history = get_indexes(words_history)
    idx2word_history_arguments = get_indexes(words_history_arguments)
    idx2word_state = get_indexes(words_state)
    idx2word_action = get_indexes(words_action)
    idx2word_action_arguments = get_indexes(words_action_arguments)
    idx2word_action_templates = get_indexes(words_action_templates, add_sos=False, add_eos=False)

    return (idx2word_history, idx2word_history_arguments,
            idx2word_state,
            idx2word_action, idx2word_action_arguments,
            idx2word_action_templates)


def get_indexes(dct, add_sos = True, add_eos=True, add_oov=True):
    idx2word = []
    if add_sos:
        idx2word.append('_SOS_')
    if add_eos:
        idx2word.append('_EOS_')
    if add_oov:
        idx2word.append('_OOV_')

    dct = [word for word in dct if dct[word] >= 2]
    idx2word.extend(sorted(dct))
    return idx2word


def index_and_pad_utterance(utterance, word2idx, max_length, add_sos=True):
    if add_sos:
        s = [word2idx['_SOS_']]
    else:
        s = []

    for w in utterance:
        # if w == '20 milton road chesterton':
        #     print(len(utterance), utterance, w, max_length)
        #     sys.exit(0)
        # if w not in word2idx:
        #     print('U', utterance)
        #     print('OOV: {oov}'.format(oov=w))
        s.append(word2idx.get(w, word2idx['_OOV_']))

    for w in range(max_length - len(s)):
        s.append(word2idx['_EOS_'])

    return s[:max_length]


def index_and_pad_history(history, word2idx, max_length_history, max_length_utterance):
    index_pad_history = []

    # padding
    for i in range(max_length_history - len(history)):
        ip_utterance = index_and_pad_utterance('', word2idx, max_length_utterance + 2)

        index_pad_history.append(ip_utterance)

    # the real data
    for utterance in history:
        ip_utterance = index_and_pad_utterance(utterance, word2idx, max_length_utterance + 2)

        index_pad_history.append(ip_utterance)

    return index_pad_history[len(index_pad_history) - max_length_history:]


def index_action_template(action_template, word2idx_action_template):
    return word2idx_action_template.get(action_template, word2idx_action_template['_OOV_'])


def index_and_pad_examples(examples,
                           word2idx_history, max_length_history, max_length_utterance,
                           word2idx_history_arguments,
                           word2idx_state, max_length_state,
                           word2idx_action, max_length_action,
                           word2idx_action_arguments,
                           word2idx_action_template):
    index_pad_examples = []
    for abs_history, history_arguments, abs_state, abs_action, action_arguments, action_template in examples:
        ip_history = index_and_pad_history(abs_history, word2idx_history, max_length_history, max_length_utterance)
        # print(len(history_arguments), history_arguments)
        ip_history_arguments = index_and_pad_utterance(history_arguments, word2idx_history_arguments,
                                                       len(history_arguments), add_sos=False)
        # print(len(ip_history_arguments), ip_history_arguments)
        ip_state = index_and_pad_utterance(abs_state, word2idx_state, max_length_state, add_sos=False)
        ip_action = index_and_pad_utterance(abs_action, word2idx_action, max_length_action, add_sos=False)
        ip_action_arguments = index_and_pad_utterance(action_arguments, word2idx_action_arguments,
                                                      len(action_arguments), add_sos=False)
        ip_action_template = index_action_template(action_template, word2idx_action_template)

        index_pad_examples.append(
                [ip_history, ip_history_arguments, ip_state, ip_action, ip_action_arguments, ip_action_template])

    return index_pad_examples


def add_action_templates(abstract_test_examples):
    examples = []
    for e in abstract_test_examples:
        examples.append(list(e) + [' '.join(e[3]), ])

    return examples


def gen_database(database_data):
    """Convert a discrete database

    :param database_data:
    :return:
    """
    idx2word = defaultdict(set)

    for row in database_data:
        for column in row:
            idx2word[column].add(row[column])
    for column in idx2word:
        idx2word[column].add('_OOV_')

    for column in idx2word:
        idx2word[column] = sorted(idx2word[column])

    word2idx = defaultdict(dict)
    for column in idx2word:
        word2idx[column] = get_word2idx(idx2word[column])

    columns = sorted(idx2word.keys())

    database = []
    for row in database_data:
        r = []
        for column in columns:
            idx = word2idx[column][row.get(column, '_OOV_')]
            r.append(idx)
        database.append(r)

    return database, columns, idx2word, word2idx

class DSTC2:
    def __init__(self, mode, input, train_data_fn, data_fraction, test_data_fn, ontology_fn, database_fn, batch_size):
        self.ontology = ontology.Ontology(ontology_fn, database_fn)

        database_data = load_json_data(database_fn)
        database, database_columns, database_idx2word, database_word2idx = gen_database(database_data)
        database = np.asarray(database, dtype=np.int32)

        train_data = load_json_data(train_data_fn)
        train_data = train_data[:int(len(train_data) * min(data_fraction, 1.0))]
        test_data = load_json_data(test_data_fn)
        test_data = test_data[:int(len(test_data) * min(data_fraction, 1.0))]

        train_examples = gen_examples(train_data, input)
        test_examples = gen_examples(test_data, input)
        # self.print_examples(train_examples

        norm_train_examples = normalize(train_examples)
        norm_train_examples = sort_by_conversation_length(norm_train_examples)
        # remove 10 % of the longest dialogues this will half the length of the conversations
        norm_train_examples = norm_train_examples[:-int(len(norm_train_examples) / 10)]
        # print(norm_train_examples)

        norm_test_examples = normalize(test_examples)
        norm_test_examples = sort_by_conversation_length(norm_test_examples)
        # remove 10 % of the longest dialogues
        norm_test_examples = norm_test_examples[:-int(len(norm_test_examples) / 10)]

        abstract_train_examples = self.ontology.abstract(norm_train_examples)
        abstract_test_examples = self.ontology.abstract(norm_test_examples)

        abstract_train_examples = add_action_templates(abstract_train_examples)
        abstract_test_examples = add_action_templates(abstract_test_examples)

        # self.print_abstract_examples(abstract_train_examples)
        #self.print_abstract_examples(abstract_test_examples)

        idx2word_history, idx2word_history_arguments, \
        idx2word_state, \
        idx2word_action, idx2word_action_arguments, \
        idx2word_action_template = get_idx2word(abstract_train_examples)
        word2idx_history = get_word2idx(idx2word_history)
        word2idx_history_arguments = get_word2idx(idx2word_history_arguments)
        word2idx_state = get_word2idx(idx2word_state)
        word2idx_action = get_word2idx(idx2word_action)
        word2idx_action_arguments = get_word2idx(idx2word_action_arguments)
        word2idx_action_template = get_word2idx(idx2word_action_template)

        # print(idx2word_history)
        # print(idx2word_history_arguments)
        # print(word2idx_state)
        # print(len(idx2word_action), idx2word_action)
        # print(len(idx2word_action_arguments), idx2word_action_arguments)
        # print(len(idx2word_action_template), idx2word_action_template)
        # print()
        # sys.exit(0)

        max_length_history = 0
        max_length_utterance = 0
        max_length_state = 0
        max_length_action = 0
        for abs_history, history_arguments, abs_state, abs_action, action_arguments, action_template in abstract_train_examples:
            for utterance in abs_history:
                max_length_utterance = max(max_length_utterance, len(utterance))

            max_length_history = max(max_length_history, len(abs_history))
            max_length_state = max(max_length_state, len(abs_state))
            max_length_action = max(max_length_action, len(abs_action))

        # pad the data with _SOS_ and _EOS_ word symbols
        train_index_examples = index_and_pad_examples(
                abstract_train_examples,
                word2idx_history, max_length_history, max_length_utterance,
                word2idx_history_arguments,
                word2idx_state, max_length_state,
                word2idx_action, max_length_action,
                word2idx_action_arguments,
                word2idx_action_template
        )
        # print(train_index_examples)
        # sys.exit(0)

        # for history, target in train_index_examples:
        #     for utterance in history:
        #         print('U', len(utterance), utterance)
        #     print('T', len(target), target)

        train_histories, \
        train_histories_arguments, \
        train_states, \
        train_actions, \
        train_actions_arguments, \
        train_actions_template = self.unpack_index_examples(train_index_examples)

        # for e in train_histories_arguments:
        #     print(len(e), e)

        # print(train_histories)
        # print(train_histories_arguments)
        # print(train_states)
        # print(train_actions)
        # print(train_actions_arguments)
        # print(train_actions_template)

        test_index_examples = index_and_pad_examples(
                abstract_test_examples,
                word2idx_history, max_length_history, max_length_utterance,
                word2idx_history_arguments,
                word2idx_state, max_length_state,
                word2idx_action, max_length_action,
                word2idx_action_arguments,
                word2idx_action_template
        )
        # print(test_index_examples)
        # sys.exit(0)

        # for history, target in test_index_examples:
        #     for utterance in history:
        #         print('U', len(utterance), utterance)
        #     print('T', len(target), target)

        test_histories, \
        test_histories_arguments, \
        test_states, \
        test_actions, \
        test_actions_arguments, \
        test_actions_template = self.unpack_index_examples(test_index_examples)

        # print(test_histories)
        # print(test_word_responses)

        train_set = {
            'histories': train_histories,
            'histories_arguments': train_histories_arguments,
            'states': train_states,
            'actions': train_actions,
            'actions_arguments': train_actions_arguments,
            'actions_template': train_actions_template
        }
        dev_set = {
            'histories': train_histories[:10:],
            'histories_arguments': train_histories_arguments[:10:],
            'states': train_states[:10:],
            'actions': train_actions[:10:],
            'actions_arguments': train_actions_arguments[:10:],
            'actions_template': train_actions_template[:10:]
        }
        test_set = {
            'histories': test_histories,
            'histories_arguments': test_histories_arguments,
            'states': test_states,
            'actions': test_actions,
            'actions_arguments': test_actions_arguments,
            'actions_template': test_actions_template
        }

        self.database = database
        self.database_columns = database_columns
        self.database_idx2word = database_idx2word
        self.database_word2idx = database_word2idx

        self.train_set = train_set
        self.train_set_size = len(train_set['histories'])
        self.dev_set = dev_set
        self.dev_set_size = len(dev_set['histories'])
        self.test_set = test_set
        self.test_set_size = len(test_set['histories'])

        self.idx2word_history = idx2word_history
        self.word2idx_history = word2idx_history
        self.idx2word_history_arguments = idx2word_history_arguments
        self.word2idx_history_arguments = word2idx_history_arguments
        self.idx2word_state = idx2word_state
        self.word2idx_state = word2idx_state
        self.idx2word_action = idx2word_action
        self.word2idx_action = word2idx_action
        self.idx2word_action_arguments = idx2word_action_arguments
        self.word2idx_action_arguments = word2idx_action_arguments
        self.idx2word_action_template = idx2word_action_template
        self.word2idx_action_template = word2idx_action_template

        self.batch_size = batch_size
        self.train_batch_indexes = [[i, i + batch_size] for i in range(0, self.train_set_size, batch_size)]

        # print(idx2word_history)
        # print(word2idx_history)
        # print()
        # print(idx2word_state)
        # print(word2idx_state)
        # sys.exit(0)

    def unpack_index_examples(self, index_examples):
        histories = np.asarray([e[0] for e in index_examples], dtype=np.int32)
        histories_arguments = np.asarray([e[1] for e in index_examples], dtype=np.int32)
        states = np.asarray([e[2] for e in index_examples], dtype=np.int32)
        actions = np.asarray([e[3] for e in index_examples], dtype=np.int32)
        actions_arguments = np.asarray([e[4] for e in index_examples], dtype=np.int32)
        actions_template = np.asarray([e[5] for e in index_examples], dtype=np.int32)

        return histories, histories_arguments, states, actions, actions_arguments, actions_template

    def iter_train_batches(self):
        for batch in self.train_batch_indexes:
            yield {
                'histories': self.train_set['histories'][batch[0]:batch[1]],
                'histories_arguments': self.train_set['histories_arguments'][batch[0]:batch[1]],
                'states': self.train_set['states'][batch[0]:batch[1]],
                'actions': self.train_set['actions'][batch[0]:batch[1]],
                'actions_arguments': self.train_set['actions_arguments'][batch[0]:batch[1]],
                'actions_template': self.train_set['actions_template'][batch[0]:batch[1]],
            }
        shuffle(self.train_batch_indexes)

    def print_examples(self, examples):
        for history, state, action in examples:
            print('-' * 120)
            for utterance in history:
                print('U', utterance)
            print('S', state)
            print('A', action)
        sys.exit(0)

    def print_abstract_examples(self, abstract_examples):
        for abs_history, history_arguments, abs_state, abs_action, action_arguments, action_template in abstract_examples:
            print('-' * 120)
            for utterance in abs_history:
                print('U    ', ' '.join(utterance))
            print('S    ', ' '.join(abs_state))
            print('ArgsH', history_arguments)
            print('A    ', ' '.join(abs_action))
            print('ArgsA', action_arguments)
        sys.exit(0)
