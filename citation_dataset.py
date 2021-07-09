import os
import tarfile
import csv
import requests
import numpy as np
import scipy.sparse as sp
#==========#==========#==========#==========#==========#==========#==========#


def load_dataset(dataset_str='cora', folder='download', self_loop_edge=False, ordar_classes='planetoid'):
    dataset = download_read_dataset(dataset_str, folder, ordar_classes)
    edges_info = show_edges_info(dataset, False)
    adj_egdes = edges_info['adj_egdes']
    true_labels = dataset['true_labels']
    features = dataset['features']
    num_nodes = len(true_labels)
    num_features = features.shape[-1]
    num_classes = int(max(true_labels) + 1)
    # adj
    list_egdes = list(adj_egdes)
    if self_loop_edge:
        loop_egdes = edges_info['loop_egdes']
        list_egdes += list(loop_egdes)
    adj = get_adj(list_egdes, num_nodes)
    # labels
    labels = sp.dok_matrix((num_nodes, num_classes))
    for i, true_label in enumerate(true_labels):
        if true_label >= 0:
            labels[i, true_label] = 1
    labels = labels.astype('float32')
    output = {
        'adj': adj,
        'features': features,
        'labels': labels.toarray(),
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': num_classes,
        'num_adj_nnz': adj.nnz,
        'num_features_nnz': features.nnz
    }
    return {**output, **dataset}


def get_adj(edges: list, num_nodes: int):
    e_rows, e_cols = np.array(edges, dtype=np.int).transpose()
    values = np.ones(shape=(len(e_rows), ), dtype=np.float32)
    adj = sp.coo_matrix((values, (e_rows, e_cols)),
                        shape=[num_nodes, num_nodes])
    # triu adj --> adj
    adj.setdiag(0)
    bigger = adj.T > adj
    adj = adj - adj.multiply(bigger) + adj.T.multiply(bigger)
    return adj
#==========#==========#==========#==========#==========#==========#==========#


def download_read_dataset(dataset_str='cora', folder='download', ordar_classes='planetoid'):
    """
    keys = ['features', 'true_labels', 'node_names', 'class_labels', 'feature_names', 'edge_names', 'edges']
    """
    assert dataset_str in ['cora', 'citeseer', 'pubmed']
    if dataset_str == 'cora':
        file1, file2 = download_cora(folder)
        data1 = read_cora_citeseer_content(file1)
        data2 = read_cora_citeseer_cites(file2)
    elif dataset_str == 'citeseer':
        file1, file2 = download_citeseer(folder)
        data1 = read_cora_citeseer_content(file1)
        data2 = read_cora_citeseer_cites(file2)
    elif dataset_str == 'pubmed':
        file1, file2 = download_pubmed(folder)
        data1 = read_pubmed_node_paper(file1)
        data2 = read_pubmed_cites(file2)
    dataset = {**data1, **data2}
    fix_lost_nodes(dataset) # for citeseer 15 lost nodes
    update_edges(dataset)
    if ordar_classes == 'planetoid':
        if dataset_str == 'cora':
            reoder_class_labels(dataset, [5, 2, 3, 4, 1, 6, 0]) # for planetoid
        elif dataset_str == 'citeseer':
            reoder_class_labels(dataset, [0, 4, 3, 5, 2, 1]) # for planetoid
        elif dataset_str == 'pubmed':
            reoder_class_labels(dataset, [0, 2, 1])
    return dataset


def fix_lost_nodes(dataset):
    node_names = dataset['node_names']
    edges = dataset['edges']
    set1 = set(node_names.tolist())
    set2 = set(edges.reshape(-1).tolist())
    lost_node_names = np.array(sorted(list(set2 - set1)))
    num_lost = len(lost_node_names)
    if num_lost == 0:
        return
    print(f'fix {num_lost} lost nodes')
    dataset['node_names'] = np.hstack([node_names, lost_node_names])
    # fix features
    features = dataset['features']
    lost_features = sp.dok_matrix(
        (num_lost, features.shape[-1])).astype('float32')
    dataset['features'] = sp.vstack([features, lost_features])
    # fix true_labels
    true_labels = dataset['true_labels']
    lost_true_labels = np.full(num_lost, -1)
    dataset['true_labels'] = np.hstack([true_labels, lost_true_labels])


def update_edges(dataset):
    node_names = dataset['node_names']
    edges = dataset['edges']
    fn = dict(zip(node_names, range(len(node_names))))
    edges = [(fn[u], fn[v]) for u, v in edges]
    dataset['edges'] = np.array(edges)
    
def reoder_class_labels(dataset, idx_origin_target:list):
    """
    idx_origin_target: orig_idx -> label -> targ_idx
    """
    class_labels = dataset['class_labels']
    true_labels = dataset['true_labels']
    assert len(class_labels) == len(idx_origin_target)
    idx_target_origin = np.argsort(idx_origin_target)
    idx_origin_target = np.argsort(idx_target_origin)
    dataset['class_labels'] = class_labels[idx_target_origin]
    if -1 in true_labels:
        temp = len(idx_origin_target)
        idx_origin_target = np.hstack([idx_origin_target, [temp]])
        true_labels = idx_origin_target[true_labels]
        true_labels[true_labels==temp] = -1
    else:
        true_labels = idx_origin_target[true_labels]
    dataset['true_labels'] = true_labels
#==========#==========#==========#==========#==========#==========#==========#


def download_file(url, folder='download'):
    # 建立 下載的資料夾
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print(f'mkdir: ./{folder}')
    file = os.path.basename(url)  # 取的 url最後的檔案名稱
    file_path = f'./{folder}/{file}'  # 加上下載的暫存路徑
    # 檢查是否已下載
    if os.path.isfile(file_path):
        print(f'existed: {file_path}')
        return file_path
    # 下載檔案
    with requests.get(url, stream=True) as r:
        with open(file_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
    print(f'download: {file_path}')
    return file_path


def download_cora(folder='download'):
    file = 'cora'
    # 檢查是否 下載並解壓縮 過
    file1 = f'./{folder}/{file}/cora.content'
    file2 = f'./{folder}/{file}/cora.cites'
    if os.path.isfile(file1) and os.path.isfile(file2):
        return file1, file2
    # 檢查是否 下載壓縮檔 過
    url = 'https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.tar.gz'
    file3 = download_file(url, folder)
    assert os.path.isfile(file3)
    # 解壓縮
    with tarfile.open(file3, "r:gz") as tar:
        tar.extractall(f'./{folder}')
    assert os.path.isfile(file1) and os.path.isfile(file2)
    print(f'unzip: {file1}')
    print(f'unzip: {file2}')
    os.remove(file3)
    os.remove(f'./{folder}/{file}/README')
    return file1, file2


def download_citeseer(folder='download'):
    file = 'citeseer-doc-classification'
    # 檢查是否 下載並解壓縮 過
    file1 = f'./{folder}/{file}/citeseer.content'
    file2 = f'./{folder}/{file}/citeseer.cites'
    if os.path.isfile(file1) and os.path.isfile(file2):
        return file1, file2
    # 檢查是否 下載壓縮檔 過
    url = 'https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.tar.gz'
    file3 = download_file(url, folder)
    assert os.path.isfile(file3)
    # 解壓縮
    with tarfile.open(file3, "r:gz") as tar:
        tar.extractall(f'./{folder}')
    assert os.path.isfile(file1) and os.path.isfile(file2)
    print(f'unzip: {file1}')
    print(f'unzip: {file2}')
    os.remove(file3)
    os.remove(f'./{folder}/{file}/README')
    return file1, file2


def download_pubmed(folder='download'):
    file = 'pubmed-diabetes'
    # 檢查是否 下載並解壓縮 過
    file1 = f'./{folder}/{file}/data/Pubmed-Diabetes.NODE.paper.tab'
    file2 = f'./{folder}/{file}/data/Pubmed-Diabetes.DIRECTED.cites.tab'
    if os.path.isfile(file1) and os.path.isfile(file2):
        return file1, file2
    # 檢查是否 下載壓縮檔 過
    url = 'https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.tar.gz'
    file3 = download_file(url, folder)
    assert os.path.isfile(file3)
    # 解壓縮
    with tarfile.open(file3, "r:gz") as tar:
        tar.extractall(f'./{folder}')
    assert os.path.isfile(file1) and os.path.isfile(file2)
    print(f'unzip: {file1}')
    print(f'unzip: {file2}')
    os.remove(file3)
    os.remove(f'./{folder}/{file}/._README')
    os.remove(f'./{folder}/{file}/README')
    os.remove(f'./{folder}/{file}/data/Pubmed-Diabetes.GRAPH.pubmed.tab')
    return file1, file2
#==========#==========#==========#==========#==========#==========#==========#


def read_cora_citeseer_content(path):
    # path = './download/cora/cora.content'
    # path = './download/citeseer-doc-classification/citeseer.content'
    data = np.genfromtxt(path, dtype=np.dtype(str))
    node_names = data[:, 0]
    features = sp.coo_matrix(data[:, 1:-1].astype('float32'))
    node_labels = data[:, -1]
    # true_labels
    class_labels = sorted(list(set(node_labels)))
    dict_class_label_to_id = dict(zip(class_labels, range(len(class_labels))))
    true_labels = [dict_class_label_to_id[i] for i in node_labels]
    return {
        'features': features,
        'true_labels': np.array(true_labels),
        'node_names': np.array(node_names),
        'class_labels': np.array(class_labels),
        'feature_names': None
    }


def read_cora_citeseer_cites(path):
    # path = './download/cora/cora.cites'
    # path = './download/citeseer-doc-classification/citeseer.cites'
    edges = np.genfromtxt(path, dtype=np.dtype(str))
    return {'edge_names': None, 'edges': edges}


def read_csv_all(path):
    data = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter='\t')
        for row in rows:
            data.append(row)
    return data


def read_pubmed_node_paper(path):
    # path='./download/pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab'
    data = read_csv_all(path)
    node_names = [row[0] for row in data[2:]]
    node_labels = [row[1].split('=')[1] for row in data[2:]]
    node_features = [row[2:-1] for row in data[2:]]
    feature_names = [field.split(':')[1] for field in data[1][1:-1]]
    # features
    features = sp.dok_matrix((len(node_names), len(feature_names)))
    dict_feature_name_to_id = dict(
        zip(feature_names, range(len(feature_names))))
    for i, row in enumerate(node_features):
        for item in row:
            key, value = item.split('=')
            j = dict_feature_name_to_id[key]
            features[i, j] = float(value)
    features = features.astype('float32')
    # true_labels
    class_labels = sorted(list(set(node_labels)))
    dict_class_label_to_id = dict(zip(class_labels, range(len(class_labels))))
    true_labels = [dict_class_label_to_id[i] for i in node_labels]
    return {
        'features': features,
        'true_labels': np.array(true_labels),
        'node_names': np.array(node_names),
        'class_labels': np.array(class_labels),
        'feature_names': np.array(feature_names)
    }


def read_pubmed_cites(path):
    # path='./download/pubmed-diabetes/data/Pubmed-Diabetes.cites.tab'
    data = read_csv_all(path)
    edge_names = [row[0] for row in data[2:]]
    cited_nodes = [row[1].split(':')[1] for row in data[2:]]
    citing_nodes = [row[3].split(':')[1] for row in data[2:]]
    edges = list(zip(cited_nodes, citing_nodes))
    return {'edge_names': np.array(edge_names), 'edges': np.array(edges)}
#==========#==========#==========#==========#==========#==========#==========#


def show_edges_info(dataset, verbose=True):
    edges = dataset['edges']
    true_labels = dataset['true_labels']
    # edges = adj_egdes + loop_egdes + undir_egdes
    adj_egdes = set()
    loop_egdes = set()
    undir_egdes = set()
    for cited, citing in edges:
        edge1 = (cited, citing)
        edge2 = (citing, cited)
        if citing == cited:
            loop_egdes.add(edge1)
        elif edge1 not in adj_egdes and edge2 not in adj_egdes:
            adj_egdes.add(edge1)
        else:
            undir_egdes.add(edge1)
    inv_undir_egdes = {(v, u) for u, v in undir_egdes}
    dir_egdes = adj_egdes - inv_undir_egdes
    if verbose:
        print(f"原始邊數 = {len(edges)}")
        print(f"自環邊數 = {len(loop_egdes)}")
        print(f"雙向邊數 = {len(undir_egdes)}")
        print(f"單向邊數 = {len(dir_egdes)}")
        print(f"無向圖邊數 = {len(adj_egdes)}")
    return {
        'adj_egdes': adj_egdes,
        'loop_egdes': loop_egdes,
        'undir_egdes': undir_egdes,
        'dir_egdes': dir_egdes
    }


def show_adj_egdes_info(dataset, adj_egdes, verbose=True):
    true_labels = dataset['true_labels']
    class_egdes = [(true_labels[u], true_labels[v]) for u, v in adj_egdes]
    num_class = len(set(true_labels))
    count_same = 0
    count_diff = 0
    count_2d = np.zeros([num_class, num_class], int)
    for u_class, v_class in class_egdes:
        if u_class < v_class:
            count_2d[u_class][v_class] += 1
            count_diff += 1
        elif u_class > v_class:
            count_2d[v_class][u_class] += 1
            count_diff += 1
        else:
            count_2d[u_class][v_class] += 1
            count_same += 1
    if verbose:
        print(f"邊的兩端節點 相同標籤 個數 = {count_same}")
        print(f"邊的兩端節點 不同標籤 個數 = {count_diff}")
        print("混淆矩陣: ")
        print(count_2d)
    return {
        'count_same': count_same,
        'count_diff': count_diff,
        'count_2d': count_2d
    }


def show_labels_info(dataset, verbose=True):
    true_labels = dataset['true_labels']
    class_labels = dataset['class_labels']
    num = len(class_labels)  # 7, 6, 3
    unique, counts = np.unique(true_labels, return_counts=True)
    dict_labels_count = dict(zip(unique, counts))
    if -1 in unique:
        index_labels_ = np.array(list(range(num)) + [-1])
        class_labels_ = np.hstack([class_labels, ['Null']])
    else:
        index_labels_ = np.array(list(range(num)))
        class_labels_ = class_labels
    count_labels_ = np.array([dict_labels_count[i] for i in index_labels_])
    if verbose:
        print(f"{'index':5}  {'class_name':22}  {'count':5}")
        for zzz in zip(index_labels_, class_labels_, count_labels_):
            i, i_class, i_count = zzz
            print(f'{i:5}  {i_class:22}  {i_count:5}')
    return {
        'index_labels_': index_labels_,
        'class_labels_': class_labels_,
        'count_labels_': count_labels_
    }


def show_dataset_info(dataset, verbose=True):
    # dataset = download('cora') # ['cora', 'citeseer', 'pubmed']
    # show edges
    edges_info = show_edges_info(dataset, verbose)
    # show adj_egdes
    adj_egdes = edges_info['adj_egdes']
    adj_egdes_info = show_adj_egdes_info(dataset, adj_egdes, verbose)
    # show labels
    labels_info = show_labels_info(dataset, verbose)
    return {**edges_info, **adj_egdes_info, **labels_info}


#==========#==========#==========#==========#==========#==========#==========#
if __name__ == '__main__':
    print(f'【cora】')
    cora = load_dataset('cora')
    cora_info = show_dataset_info(cora)
    all_keys = list(cora.keys()) + list(cora_info.keys())
    print(f'all_keys = {all_keys}')

    print(f'\n【citeseer】')
    citeseer = load_dataset('citeseer')
    citeseer_info = show_dataset_info(citeseer)
    all_keys = list(citeseer.keys()) + list(citeseer_info.keys())
    print(f'all_keys = {all_keys}')

    print(f'\n【pubmed】')
    pubmed = load_dataset('pubmed')
    pubmed_info = show_dataset_info(pubmed)
    all_keys = list(pubmed.keys()) + list(pubmed_info.keys())
    print(f'all_keys = {all_keys}')
