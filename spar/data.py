"""SPAR SEED/SEED-IV DE-LDS loading and normalization."""
from __future__ import annotations
import os
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset

NUM_CHANNELS, NUM_BANDS, FEATURE_DIM = 62, 5, 310
# Set explicitly through the CLI; no private machine paths are embedded.
dataset_path = {}

def flatten_de_lds(raw, feature_layout="legacy_channel_major", preserve_dtype=False):
    if feature_layout != "legacy_channel_major":
        raise ValueError("Only the main-method channel-major layout is supported")
    validate_de_lds_feature(raw)
    flat = raw.transpose(1, 2, 0).reshape(raw.shape[1], 310, order="F")
    return flat if preserve_dtype else flat.astype(np.float32, copy=False)

def norminy(data):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i])
    return dataT.T

def normalization(data):
    '''
    description: 
    param {type} 
    return {type} 
    '''
    _range = np.max(data) - np.min(data)
    if _range == 0:
        return np.zeros_like(data)
    return (data - np.min(data)) / _range

# package the data and label into one class

class CustomDataset(Dataset):
    # initialization: data and label
    def __init__(
        self,
        Data,
        Label,
        sample_indices=None,
        return_index=False,
        trial_ids=None,
        window_ids=None,
    ):
        self.Data = Data
        self.Label = Label
        self.return_index = bool(return_index)
        if sample_indices is None:
            sample_indices = np.arange(len(Data), dtype=np.int64)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        if len(self.sample_indices) != len(Data):
            raise ValueError('sample_indices length must match data length')
        self.trial_ids = (
            None if trial_ids is None else np.asarray(trial_ids, dtype=np.int64)
        )
        self.window_ids = (
            None if window_ids is None else np.asarray(window_ids, dtype=np.int64)
        )
        if self.trial_ids is not None and len(self.trial_ids) != len(Data):
            raise ValueError('trial_ids length must match data length')
        if self.window_ids is not None and len(self.window_ids) != len(Data):
            raise ValueError('window_ids length must match data length')
    # get the size of data

    def __len__(self):
        return len(self.Data)
    # get the data and label

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        if self.return_index:
            return data, label, torch.tensor(self.sample_indices[index], dtype=torch.int64)
        return data, label


def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type} 
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variables
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]
    #SEED3:labels (-1 for negative, 0 for neutral and +1 for positive)
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')

def sort_numerically_by_prefix(filenames):
    def sort_key(name):
        stem = os.path.splitext(os.path.basename(name))[0]
        prefix = stem.split('_', 1)[0]
        if prefix.isdigit():
            return (0, int(prefix), stem)
        return (1, stem)

    return sorted(filenames, key=sort_key)

def sort_de_lds_keys(keys):
    def sort_key(name):
        suffix = name.replace('de_LDS', '')
        if suffix.isdigit():
            return int(suffix)
        return suffix

    return sorted(keys, key=sort_key)

def validate_de_lds_feature(feature, mat_path=None, key=None):
    if feature.ndim != 3 or feature.shape[0] != NUM_CHANNELS or feature.shape[2] != NUM_BANDS:
        location = ''
        if mat_path is not None:
            location += ' file={}'.format(mat_path)
        if key is not None:
            location += ' key={}'.format(key)
        raise ValueError(
            'Expected de_LDS feature shape (62, N, 5), got {}.{}'.format(
                feature.shape,
                location,
            )
        )

def reshape_data(
    data,
    label,
    mat_path=None,
    keys=None,
    feature_layout="legacy_channel_major",
    preserve_dtype=False,
    return_metadata=False,
):
    '''
    description: reshape data and initiate corresponding label vectors
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*310
        reshape_label: array, x*1
    '''
    reshape_data = None
    reshape_label = None
    trial_ids = []
    window_ids = []

    for i in range(len(data)):
        key = keys[i] if keys is not None else None
        validate_de_lds_feature(data[i], mat_path=mat_path, key=key)
        one_data = flatten_de_lds(
            data[i],
            feature_layout=feature_layout,
            preserve_dtype=preserve_dtype,
        )
        if one_data.ndim != 2 or one_data.shape[1] != FEATURE_DIM:
            raise ValueError(
                'Flattened de_LDS feature expected shape [N, 310], got {} file={} key={}'.format(
                    one_data.shape,
                    mat_path,
                    key,
                )
            )
        one_label = np.full((one_data.shape[0], 1), label[i])
        if return_metadata:
            trial_ids.append(np.full(one_data.shape[0], i, dtype=np.int64))
            window_ids.append(np.arange(one_data.shape[0], dtype=np.int64))
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    if return_metadata:
        metadata = {
            'trial_ids': np.concatenate(trial_ids),
            'window_ids': np.concatenate(window_ids),
        }
        return reshape_data, reshape_label, metadata
    return reshape_data, reshape_label

def get_data_label_frommat(
    mat_path,
    dataset_name,
    session_id,
    feature_layout="legacy_channel_major",
    preserve_dtype=False,
    return_metadata=False,
):
    '''
    description: load data from mat path and reshape to 851*310
    param {type}:
        mat_path: String
        session_id: int
    return {type}: 
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''
    _, _, labels = get_number_of_label_n_trial(dataset_name)

    mat_data = scio.loadmat(mat_path)
    mat_keys = sort_de_lds_keys([key for key in mat_data.keys() if key.startswith('de_LDS')])
    if len(mat_keys) != len(labels[session_id]):
        raise ValueError(
            'Expected {} de_LDS keys in {}, got {}'.format(
                len(labels[session_id]), mat_path, len(mat_keys)
            )
        )
    print(
        'load {} key_count {} first_key {} last_key {}'.format(
            os.path.basename(mat_path),
            len(mat_keys),
            mat_keys[0],
            mat_keys[-1],
        )
    )
    mat_de_data = []
    for key in mat_keys:
        validate_de_lds_feature(mat_data[key], mat_path=mat_path, key=key)
        mat_de_data.append(mat_data[key])

    reshaped = reshape_data(
        mat_de_data,
        labels[session_id],
        mat_path=mat_path,
        keys=mat_keys,
        feature_layout=feature_layout,
        preserve_dtype=preserve_dtype,
        return_metadata=return_metadata,
    )
    return reshaped

def get_allmats_name(dataset_name):
    '''
    description: get the names of all the .mat files
    param {type}
    return {type}:
        allmats: list (3*15)
    '''
    path = dataset_path[dataset_name]
    entries = sorted(os.listdir(path))
    session_dirs = [
        entry for entry in entries
        if entry != '.DS_Store' and os.path.isdir(os.path.join(path, entry))
    ]

    if session_dirs:
        allmats = []
        sorted_session_dirs = sorted(session_dirs, key=lambda session: int(session) if session.isdigit() else session)
        for session in sorted_session_dirs:
            mats = sort_numerically_by_prefix(
                mat for mat in os.listdir(os.path.join(path, session))
                if mat.endswith('.mat') and mat != 'label.mat'
            )
            print('session {} files: {}'.format(session, mats))
            allmats.append([os.path.join(path, session, mat) for mat in mats])
        return allmats

    flat_mats = [
        entry for entry in entries
        if entry.endswith('.mat') and entry != 'label.mat'
    ]
    mats_by_subject = {}
    for mat in flat_mats:
        subject_token = mat.split('_', 1)[0]
        if subject_token.isdigit():
            subject_id = int(subject_token)
            mats_by_subject.setdefault(subject_id, []).append(mat)

    allmats = [[] for _ in range(3)]
    for subject_id in sorted(mats_by_subject):
        subject_mats = sort_numerically_by_prefix(mats_by_subject[subject_id])
        if len(subject_mats) != 3:
            raise ValueError(
                'Expected 3 session files for subject {}, got {}'.format(
                    subject_id, len(subject_mats)
                )
            )
        for session_idx, mat in enumerate(subject_mats):
            allmats[session_idx].append(os.path.join(path, mat))

    if any(len(session) != 15 for session in allmats):
        raise ValueError(
            'Expected 15 subjects per session, got {}'.format(
                [len(session) for session in allmats]
            )
        )

    for session_idx, session_mats in enumerate(allmats, start=1):
        print('session {} files: {}'.format(session_idx, [os.path.basename(item) for item in session_mats]))

    return allmats

def load_data(
    dataset_name,
    feature_layout="legacy_channel_major",
    preserve_dtype=False,
    return_metadata=False,
):
    '''
    description: get all the data from one dataset
    param {type} 
    return {type}:
        data: list 3(sessions) * 15(subjects), each data is x * 310
        label: list 3*15, x*1
    '''
    allmats = get_allmats_name(dataset_name)
    if len(allmats) != 3 or any(len(session) != 15 for session in allmats):
        raise ValueError('Expected exactly 3 sessions with 15 feature files each')
    data = [([0] * 15) for i in range(3)]
    label = [([0] * 15) for i in range(3)]
    metadata = [([None] * 15) for i in range(3)] if return_metadata else None
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = allmats[i][j]
            loaded = get_data_label_frommat(
                mat_path,
                dataset_name,
                i,
                feature_layout=feature_layout,
                preserve_dtype=preserve_dtype,
                return_metadata=return_metadata,
            )
            if return_metadata:
                one_data, one_label, one_metadata = loaded
                metadata[i][j] = {
                    'trial_ids': one_metadata['trial_ids'].copy(),
                    'window_ids': one_metadata['window_ids'].copy(),
                }
            else:
                one_data, one_label = loaded
            data[i][j] = one_data.copy()
            label[i][j] = one_label.copy()
    if return_metadata:
        return data, label, metadata
    return data, label
