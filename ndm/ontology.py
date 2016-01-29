#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from random import randint

from tfx.logging import LogMessage

debug = False


# debug = True


class Arguments:
    def __init__(self, n_arguments_per_slot, slots):
        self.n_arguments_per_slot = n_arguments_per_slot
        self.value2argument = {}
        self.argument2value = {}
        self.slots = set(slots)

    def __str__(self):
        return str(self.argument2value)

    def items(self):
        if self.n_arguments_per_slot > 1:
            i = []
            for slot in sorted(self.slots):
                for n in range(self.n_arguments_per_slot):
                    ARGUMENT = slot + '_' + str(n)

                    if ARGUMENT in self.argument2value:
                        i.append((ARGUMENT, self.argument2value[ARGUMENT]))
                    else:
                        i.append((ARGUMENT, 'none'))
        else:
            i = []
            for slot in sorted(self.slots):
                ARGUMENT = slot + '_X'
                if ARGUMENT in self.argument2value:
                    i.append((ARGUMENT, self.argument2value[ARGUMENT]))
                else:
                    i.append((ARGUMENT, 'none'))

        return i

    def values(self):
        return [i[1] for i in self.items()]

    def add(self, value, argument):
        self.value2argument[value] = argument
        self.argument2value[argument] = value

    def get_argument(self, value, slot):
        if value in self.value2argument:
            return self.value2argument[value]
        else:
            # I must add the argument and the return it
            if self.n_arguments_per_slot > 1:
                for i in range(0, self.n_arguments_per_slot * 3):
                    n = randint(0, self.n_arguments_per_slot - 1)
                    ARGUMENT = slot + '_' + str(n)

                    if ARGUMENT in self.argument2value:
                        continue
                    else:
                        self.add(value, ARGUMENT)
                        return ARGUMENT

                # overwrite an existing argument !!!
                m = LogMessage()
                m.add('-' * 120)
                m.add('WARNING:   overwriting an existing argument ARGUMENT = {a}, VALUE = {v}'.format(a=ARGUMENT,
                                                                                                       v=value))
                m.add('ARGUMENTS: {a}'.format(a=self.argument2value))
                m.add('')
                m.log()
                self.add(value, ARGUMENT)
            else:
                # there is only one argument per slot
                ARGUMENT = slot + '_X'
                self.add(value, ARGUMENT)

            return ARGUMENT


class Ontology:
    def __init__(self, ontology_fn, database_fn):
        self.slots = set()
        self.load_ontology(ontology_fn, database_fn)

    def load_ontology(self, ontology_fn, database_fn):
        """Load ontology - a json file. And extend it with values from teh database.

        :param ontology_fn: a name of the ontology json file
        :param database_fn: a name of the database json file
        """
        with open(ontology_fn) as f:
            ontology = json.load(f)

        with open(database_fn) as f:
            database = json.load(f)

        # slot-value-form mapping
        self.svf = defaultdict(lambda: defaultdict(set))
        # form-value-slot mapping
        self.fvs = defaultdict(lambda: defaultdict(set))
        # form-value mapping
        self.fv = {}
        # form-slot mapping
        self.fs = {}
        for slot in ontology['informable']:
            for value in ontology['informable'][slot]:
                self.add_entity(slot, value, value)

        for entity in database:
            for slot in entity:
                value = entity[slot]

                self.add_entity(slot, value, value)

        # HACK: add extra surface forms alternatives
        self.add_entity('PRICERANGE', 'moderate', 'moderately')
        self.add_entity('PRICERANGE', 'moderate', 'modreately')
        self.add_entity('PRICERANGE', 'expensive', 'spensive')
        self.add_entity('FOOD', 'asian oriental', 'asian')
        self.add_entity('FOOD', 'mediterranean', 'mediteranian')
        self.add_entity('FOOD', 'barbeque', 'barbecue')
        self.add_entity('FOOD', 'cantonese', 'cantonates')
        # self.add_entity('FOOD', '', '')

    def add_entity(self, slot, value, form):
        self.slots.add(slot.upper())
        self.svf[slot][value].add(tuple(form.lower().split()))
        self.fvs[tuple(form.lower().split())][value].add(slot)
        self.fv[tuple(form.lower().split())] = value
        self.fs[tuple(form.lower().split())] = slot.upper()

    def abstract_utterance_helper(self, init_i, utterance, history_arguments):
        for i in range(init_i, len(utterance)):
            for j in range(len(utterance), i, -1):
                slice = tuple(utterance[i:j])

                # print('s', slice)
                if slice in self.fvs:
                    value = self.fv[slice]
                    # print('f', value)

                    # skip this common false positive
                    if i > 5:
                        # print(utterance[i-1:j+1])
                        if utterance[i - 1:j + 1] == ['can', 'ask', 'for']:
                            continue

                    ARGUMENT = [history_arguments.get_argument(value, self.fs[slice])]

                    return i + 1, list(utterance[:i]) + ARGUMENT + list(utterance[j:]), True

        return len(utterance), utterance, False

    def abstract_utterance(self, utterance, history_arguments):
        abs_utt = utterance

        changed = True
        i = 0
        while changed:
            i, abs_utt, changed = self.abstract_utterance_helper(i, abs_utt, history_arguments)
            # if changed:
            #     print('au', abs_utt)
        return abs_utt

    def abstract_target(self, target, history_arguments, mode):
        if mode == 'tracker':
            return self.abstract_utterance(target, history_arguments)
        else:  # mode == 'e2e'
            return self.abstract_utterance(target, history_arguments)

    def abstract(self, examples):
        abstract_examples = []

        for history, state, action in examples:
            if debug:
                print('=' * 120)
            abs_history = []
            history_arguments = Arguments(n_arguments_per_slot=5, slots=self.slots)
            for utterance in history:
                abs_utt = self.abstract_utterance(utterance, history_arguments)
                abs_history.append(abs_utt)

                if debug:
                    print('U    ', utterance)
                    print('AbsU ', abs_utt)
                    print('ArgsH', history_arguments)
                    print()

            # if '' in history_arguments.values():
            #     print('U    ', utterance)
            #     print('AbsU ', abs_utt)
            #     print('ArgsH', history_arguments)
            #     print()
            #     sys.exit(0)

            abs_state = self.abstract_utterance(state, history_arguments)
            history_arguments = history_arguments.values()

            action_arguments = Arguments(n_arguments_per_slot=1, slots=self.slots)
            abs_action = self.abstract_utterance(action, action_arguments)
            action_arguments = action_arguments.values()

            if debug:
                print('S    ', state)
                print('AbsS ', abs_state)
                print('A    ', action)
                print('AbsA ', abs_action)
                print('ArgsA', action_arguments)
                print()

            abstract_examples.append([abs_history, history_arguments, abs_state, abs_action, action_arguments])

        return abstract_examples
