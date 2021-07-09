import os
import sys
import requests
import collections
import random
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
#==========#==========#==========#==========#==========#==========#==========#


def load_planetoid(dataset_str='cora', folder='download', verbose=False, self_loop_edge=False):
    # ['cora', 'citeseer', 'pubmed']
    dataset = download_read_planetoid(dataset_str, None, folder, verbose)
    allx, tx = dataset['allx'], dataset['tx']
    ally, ty = dataset['ally'], dataset['ty']
    test_idx_reorder = dataset['test.index']
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        num_test_extended = len(test_idx_range_full)  # num=1015
        test_extended_index = test_idx_range - min(test_idx_range)
        # Fix tx
        tx_extended = sp.lil_matrix((num_test_extended, tx.shape[1]))
        tx_extended[test_extended_index, :] = tx
        tx = tx_extended
        # Fix ty
        ty_extended = np.zeros((num_test_extended, ty.shape[1]))
        ty_extended[test_extended_index, :] = ty
        ty = ty_extended
    # features
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.astype('float32')
    # labels
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels.astype('float32')  # citeseer 15 num 0-vector
    true_labels = one_hot_labels_to_true_labels(labels, verbose=verbose)
    # adj
    graph = dataset['graph']
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.astype('float32')
    if self_loop_edge:
        adj = adj.tolil()
        adj.setdiag(0.0)
    num_nodes = features.shape[0]
    num_features = features.shape[-1]
    num_classes = labels.shape[-1]
    if dataset_str == 'cora':
        class_labels = ['Theory', 'Reinforcement_Learning', 'Genetic_Algorithms',
                        'Neural_Networks', 'Probabilistic_Methods', 'Case_Based', 'Rule_Learning']
    elif dataset_str == 'citeseer':
        class_labels = ['AI', 'ML', 'IR', 'DB', 'Agents', 'HCI']
    elif dataset_str == 'pubmed':
        class_labels = ['1', '3', '2']
    else:
        class_labels = np.arange(num_classes).astype(np.str)
    output = {
        'adj': adj,
        'features': features,
        'labels': labels,
        'true_labels': true_labels,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': num_classes,
        'num_adj_nnz': adj.nnz,
        'num_features_nnz': features.nnz,
        'node_names': np.arange(num_nodes).astype(np.str),
        'class_labels': class_labels,
        'feature_names': None,
        'edge_names': None,
        'edges': None
    }
    return {**output, **dataset}


def one_hot_labels_to_true_labels(labels, new_label=None, verbose=False):
    true_labels = labels.argmax(-1)
    # 檢查 all zero label indexs
    ids = np.where(labels.sum(-1) != 1)[0]
    if new_label is None:
        new_label = -1
    if len(ids) > 0 and verbose:
        print(f'other zero label = {ids}', end=', ')
        print(f'set new_label = {new_label}')
    for i in ids:
        true_labels[i] = new_label
    return true_labels


def download_read_planetoid(dataset_str='cora', names=None, folder='download', verbose=True):
    dict_planetoid = download_planetoid(dataset_str, names, folder)
    for name, path in dict_planetoid.items():
        if name == 'test.index':
            dict_planetoid[name] = read_test_index_file(path, verbose)
        else:
            dict_planetoid[name] = read_pkl_file(path, verbose)
    return dict_planetoid

#==========#==========#==========#==========#==========#==========#==========#


def download_file(url, folder='download'):
    # 建立 下載的資料夾
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print(f'mkdir: ./{folder}')
    file = os.path.basename(url)  # 取的 url最後的檔案名稱
    file_path = os.path.join(*[folder, file])  # 加上下載的暫存路徑
    # 檢查是否已下載
    if os.path.isfile(file_path):
        #print(f'existed: {file_path}')
        return file_path
    # 下載檔案
    with requests.get(url, stream=True) as r:
        with open(file_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
    print(f'download: {file_path}')
    return file_path


def download_planetoid(dataset_str='cora', names=None, folder='download'):
    assert dataset_str in ['cora', 'citeseer', 'pubmed']
    names_ = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    assert names is None or set(names).issubset(names_)
    if names is None:
        names = names_
    # 下載 檔案
    files = [f'ind.{dataset_str}.{name}' for name in names]
    file_paths = [f'{folder}/{file}' for file in files]
    for file in files:
        url = f'https://github.com/kimiyoung/planetoid/raw/master/data/{file}'
        download_file(url, folder)
    return dict(zip(names, file_paths))
#==========#==========#==========#==========#==========#==========#==========#


def read_pkl_file(file_path, verbose=True):
    # ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph']
    with open(file_path, 'rb') as f:
        if sys.version_info > (3, 0):
            obj = pkl.load(f, encoding='latin1')
        else:
            obj = pkl.load(f)
    if verbose:
        print(f"Load: {file_path}")
    return obj


def read_test_index_file(file_path, verbose=True):
    # ['test.index']
    index = np.genfromtxt(file_path, dtype=np.dtype(int)).tolist()
    if verbose:
        imin, imax = min(index), max(index)
        idff = imax - imin
        print(f"Load: {file_path}  ", end='')
        print(f"index: len={len(index)}, min={imin}, max={imax}, diff={idff}")
    return index
#==========#==========#==========#==========#==========#==========#==========#


def sample_mask(idx, shape):
    mask = np.zeros(shape)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_labels_mask(dataset, verbose=False):
    labels = dataset['labels']
    x = dataset['x']
    y = dataset['y']
    test_idx_reorder = dataset['test.index']
    test_idx_range = np.sort(test_idx_reorder)
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    if verbose:
        print(f"  y_train: shape={y_train.shape}")
        print(f"  y_val:   shape={y_val.shape}")
        print(f"  y_test:  shape={y_test.shape}")
        print(f"  train_mask: shape={train_mask.shape}")
        print(f"  val_mask:   shape={val_mask.shape}")
        print(f"  test_mask:  shape={test_mask.shape}")
    return {
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }


def random_labels_mask(dataset,
                       num_train_per_class: int = 20,
                       num_val: int = 500,
                       num_test: int = 1000,
                       verbose=False):
    labels = dataset['labels']
    true_labels = dataset['true_labels']
    num_nodes, num_classes = labels.shape

    dict_labels_to_nodes_list = collections.defaultdict(list)
    for i, true_label in enumerate(true_labels):
        dict_labels_to_nodes_list[true_label].append(i)

    # train: num_classes * 20
    train_nodes = []
    for i_class in range(num_classes):
        train_nodes += random.sample(dict_labels_to_nodes_list[i_class],
                                     num_train_per_class)
    train_nodes = sorted(train_nodes)
    # val:  500
    other_nodes = set(range(num_nodes)) - set(train_nodes)
    val_nodes = sorted(random.sample(other_nodes, num_val))
    # test: 1000
    other_nodes -= set(val_nodes)
    test_nodes = sorted(random.sample(other_nodes, num_test))
    # other:
    other_nodes -= set(test_nodes)
    # mask
    train_mask = sample_mask(train_nodes, num_nodes)
    val_mask = sample_mask(val_nodes, num_nodes)
    test_mask = sample_mask(test_nodes, num_nodes)
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    if verbose:
        print(f"  y_train: shape={y_train.shape}")
        print(f"  y_val:   shape={y_val.shape}")
        print(f"  y_test:  shape={y_test.shape}")
        print(f"  train_mask: shape={train_mask.shape}")
        print(f"  val_mask:   shape={val_mask.shape}")
        print(f"  test_mask:  shape={test_mask.shape}")
    return {
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }


def get_labels_mask(dataset,
                    split: str = 'public',
                    num_train_per_class: int = 20,
                    num_val: int = 500,
                    num_test: int = 1000,
                    verbose=False):
    assert split in ['public', 'random']
    if verbose:
        print(f'select {split} split to labels_mask')
    if split == 'public':
        return load_labels_mask(dataset, verbose)
    else:
        return random_labels_mask(dataset, num_train_per_class, num_val, num_test, verbose)


#==========#==========#==========#==========#==========#==========#==========#
if __name__ == '__main__':
    # 'cora', 'citeseer', 'pubmed'
    cora = load_planetoid('cora', verbose=True)
    citeseer = load_planetoid('citeseer', verbose=True)
    pubmed = load_planetoid('pubmed', verbose=True)
    # 'public', 'random'
    dataset = cora
    public_labels_mask = get_labels_mask(dataset, split='public', verbose=True)
    random_labels_mask = get_labels_mask(dataset, split='random', verbose=True)
