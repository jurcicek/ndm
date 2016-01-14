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


def gen_examples(text_data):
    """Generates training examples for the conversation to sequence model. Here, we experiment with conversational models
    that is converting a conversational history into dialogue state representation (dialogue state tracking) and generation
    a textual response given the conversation history (dialogue policy).

    :param text_data: a list of conversation each composed of (system_output, user_input, dialogue state) tuples
    :return: a transformed text_data
    """
    examples = []
    for conversation in text_data:
        history = []
        prev_turn = None
        for turn in conversation:
            if prev_turn:
                history.append(prev_turn[0])
                history.append(prev_turn[1])
                state = prev_turn[2]  # the dialogue state
                action = turn[0]  # the system action / response
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
    idx2word_action_templates = get_indexes(words_action_templates)

    return (idx2word_history, idx2word_history_arguments,
            idx2word_state,
            idx2word_action, idx2word_action_arguments,
            idx2word_action_templates)


def get_indexes(dct):
    idx2word_history = ['_SOS_', '_EOS_', '_OOV_']
    dct = [word for word in dct if dct[word] >= 2]
    idx2word_history.extend(sorted(dct))
    return idx2word_history


def index_and_pad_utterance(utterance, word2idx, max_length, add_sos=True):
    if add_sos:
        s = [word2idx['_SOS_']]
    else:
        s = []

    for w in utterance:
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
    return [word2idx_action_template.get(action_template, word2idx_action_template['_OOV_']),]


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
        ip_history_arguments = index_and_pad_utterance(history_arguments, word2idx_history_arguments, len(history_arguments[0]), add_sos=False)
        ip_state = index_and_pad_utterance(abs_state, word2idx_state, max_length_state, add_sos=False)
        ip_action = index_and_pad_utterance(abs_action, word2idx_action, max_length_action, add_sos=False)
        ip_action_arguments = index_and_pad_utterance(action_arguments, word2idx_action_arguments, len(action_arguments[0]), add_sos=False)
        ip_action_template = index_action_template(action_template, word2idx_action_template)

        index_pad_examples.append([ip_history, ip_history_arguments, ip_state, ip_action, ip_action_arguments, ip_action_template])

    return index_pad_examples


def add_action_templates(abstract_test_examples):
    examples = []
    for e in abstract_test_examples:
        examples.append(list(e) + [' '.join(e[3]), ])

    return examples


class DSTC2:
    def __init__(self, mode, train_data_fn, data_fraction, test_data_fn, ontology_fn, database_fn, batch_size):
        self.ontology = ontology.Ontology(ontology_fn, database_fn)

        train_data = load_json_data(train_data_fn)
        train_data = train_data[:int(len(train_data) * min(data_fraction, 1.0))]
        test_data = load_json_data(test_data_fn)
        test_data = test_data[:int(len(test_data) * min(data_fraction, 1.0))]

        train_examples = gen_examples(train_data)
        test_examples = gen_examples(test_data)
        # self.print_examples(train_examples)

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
        # self.print_abstract_examples(abstract_test_examples)

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

        # print(word2idx_history)
        # print(idx2word_history)
        print(len(idx2word_action), idx2word_action)
        print(len(idx2word_action_arguments), idx2word_action_arguments)
        print(len(idx2word_action_template), idx2word_action_template)
        # print()
        sys.exit(0)

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

        train_histories = [e[0] for e in train_index_examples]
        train_histories_arguments = [e[1] for e in train_index_examples]
        train_states = [e[2] for e in train_index_examples]
        train_actions = [e[3] for e in train_index_examples]
        train_actions_arguments = [e[4] for e in train_index_examples]
        train_actions_template = [e[5] for e in train_index_examples]

        train_histories = np.asarray(train_histories, dtype=np.int32)
        train_histories_arguments = np.asarray(train_histories_arguments, dtype=np.int32)
        train_states = np.asarray(train_states, dtype=np.int32)
        train_actions = np.asarray(train_actions, dtype=np.int32)
        train_actions_arguments = np.asarray(train_actions_arguments, dtype=np.int32)
        train_actions_template = np.asarray(train_actions_template, dtype=np.int32)

        print(train_histories)
        print(train_states)
        print(train_actions_template)


        test_index_examples = index_and_pad_examples(
                abstract_test_examples, word2idx_history, max_length_history, max_length_utterance,
                word2idx_target, max_length_state
        )
        # print(test_index_examples)
        # sys.exit(0)

        # for history, target in test_index_examples:
        #     for utterance in history:
        #         print('U', len(utterance), utterance)
        #     print('T', len(target), target)

        test_histories = [history for history, _ in test_index_examples]
        test_word_responses = [target for _, target in test_index_examples]

        test_histories = np.asarray(test_histories, dtype=np.int32)
        test_word_responses = np.asarray(test_word_responses, dtype=np.int32)

        # print(test_histories)
        # print(test_word_responses)

        train_set = {
            'histories':      train_histories,
            'word_responses': train_states
        }
        dev_set = {
            'histories': train_histories          [int(len(train_set) * 0.9):],
            'word_responses': train_states[int(len(train_set) * 0.9):]
        }
        test_set = {
            'histories':      test_histories,
            'word_responses': test_word_responses
        }

        self.train_set = train_set
        self.train_set_size = len(train_set['histories'])
        self.dev_set = dev_set
        self.dev_set_size = len(dev_set['histories'])
        self.test_set = test_set
        self.test_set_size = len(test_set['histories'])

        self.idx2word_history = idx2word_history
        self.word2idx_history = word2idx_history
        self.idx2word_target = idx2word_state
        self.word2idx_target = word2idx_target

        self.batch_size = batch_size
        self.train_batch_indexes = [[i, i + batch_size] for i in range(0, self.train_set_size, batch_size)]

        # print(idx2word_history)
        # print(word2idx_history)
        # print()
        # print(idx2word_target)
        # print(word2idx_target)
        # sys.exit(0)

    def iter_train_batches(self):
        for batch in self.train_batch_indexes:
            yield {
                'histories': self.train_set['histories']          [batch[0]:batch[1]],
                'word_responses': self.train_set['word_responses'][batch[0]:batch[1]],
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
