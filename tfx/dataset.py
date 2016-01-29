#!/usr/bin/env python3
import h5py


def create_dataset(file, name, data, dtype, labels):
    dataset = file.create_dataset(
            name,
            (
                data.shape[0],
                data.shape[1],
            ),
            dtype='float32'
    )
    dataset.dims[0].label = labels[0]
    dataset.dims[1].label = labels[1]

    dataset[...] = data


def save_dataset(fn,
                 train_features, train_targets,
                 valid_features, valid_targets,
                 test_features, test_targets):
    f = h5py.File(fn, mode='w')

    create_dataset(f, train_features, name='train_features', dtype='float32', labels=['example', 'feature'])
    create_dataset(f, train_targets, name='train_targets', dtype='uint8', labels=['example', 'label'])

    create_dataset(f, valid_features, name='valid_features', dtype='float32', labels=['example', 'feature'])
    create_dataset(f, valid_targets, name='valid_targets', dtype='uint8', labels=['example', 'label'])

    create_dataset(f, test_features, name='test_features', dtype='float32', labels=['example', 'feature'])
    create_dataset(f, test_targets, name='test_targets', dtype='uint8', labels=['example', 'label'])

    f.flush()
    f.close()
