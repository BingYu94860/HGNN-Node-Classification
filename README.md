# HGNN-Node-Classification

環境：

1. 使用 Tensorflow 2.x
2. 在 Anaconda 的 Jupyter Notebook 上執行
3. 亦可以在 Colab上運行

```python
#### 掛接 Google 雲端硬碟 ####
from google.colab import drive
drive.mount('/content/drive')

####　切換工作資料夾的目錄 ####
# root 需要更改成 自己的 資料夾路徑，並資料夾內放 所有 *.py 與 *.ipynb 的檔案，以及 utils 的資料夾。
root = "/content/drive/MyDrive/Colab Notebooks/HGNN-Node-Classification"
import os
os.chdir(root)
```



###### 資料集：(planetoid_dataset.py)  # 模型主要運行的檔案

1. cora
2. citeseer
3. pubmed

會自動下載 小行星planetoid： https://github.com/kimiyoung/planetoid/raw/master/data/  (對應 小行星planetoid 的 paper https://arxiv.org/abs/1603.08861 )

而這篇 [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) 的 Github 也適用使用相同內容的檔案  https://github.com/tkipf/gcn/tree/master/gcn/data

以 cora 的資料集為例，執行 gd = load_planetoid('cora') 後，就會**自動下載**以下8個檔案，就會下載至該目錄的 ./download/ 內。

1. ind.cora.x
2. ind.cora.tx
3. ind.cora.allx
4. ind.cora.y
5. ind.cora.ty
6. ind.cora.ally
7. ind.cora.graph
8. ind.cora.test.index



###### 資料集：(citation_dataset.py)  # 分析讀取 最原始的資料集 

1. cora ( https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.tar.gz )
2. citeseer ( https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.tar.gz )
3. pubmed ( https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.tar.gz )

以 cora 的資料集為例 就會**自動下載**以下8個檔案，執行 gd = load_dataset('cora') 後，就會下載並解壓縮至該目錄的 ./download/ 內。

- ./download/cora/cora.cites
- ./download/cora/cora.content

而 citeseer的資料集就會下載並解壓縮成

- ./download/citeseer-doc-classification/citeseer.cites 
- ./download/citeseer-doc-classification/citeseer.content

而 pubmed 的資料集就會下載並解壓縮成

- ./download/pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab
- ./download/pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab

※ 備註: planetoid_dataset.py 與 citation_dataset.py 所得 節點的排序是不一樣的。

※ 備註: 而標籤的順序  citation_dataset.py 會排序與 planetoid_dataset.py 一樣的順序。



# Part1 展示 planetoid_dataset.py 內的使用方法

執行「Show planetoid_dataset.ipynb」的 Jupyter Notebook的檔案，需要有以下「./planetoid_dataset.py 」結構的檔案。



# Part2 展示 citation_dataset.py 內的使用方法，並顯示相關統計數據

執行「Show citation_dataset.ipynb」的 Jupyter Notebook的檔案，需要有以下「./citation_dataset.py 」結構的檔案。



# Part3 查看 GCN Model 的執行結果 

執行「Run GCN Model.ipynb」的 Jupyter Notebook的檔案，需要有以下「./planetoid_dataset.py 、./utils/tsne.py 、./utils/norm.py 、./utils/convert.py」結構的檔案。



# Part4 查看 Chebyshev GCN Model 的執行結果

執行「Run Chebyshev GCN Model.ipynb」的 Jupyter Notebook的檔案，需要有以下「./planetoid_dataset.py 、./utils/tsne.py 、./utils/norm.py 、./utils/convert.py」結構的檔案。



# Part5 查看 HGNN Model (使用A+I來建立H) 的執行結果

執行「Run HGNN Model (A+I).ipynb」的 Jupyter Notebook的檔案，需要有以下「./planetoid_dataset.py 、./utils/tsne.py 、./utils/norm.py 、./utils/convert.py」結構的檔案。



# Part6 查看 HGNN Model (使用無向邊來建立超邊) 的執行結果

執行「Run HGNN Model (Edge).ipynb」的 Jupyter Notebook的檔案，需要有以下「./planetoid_dataset.py 、./utils/tsne.py 、./utils/norm.py 、./utils/convert.py」結構的檔案。



# Part7 查看 HGNN Model (使用最近n的鄰居來建立超邊) 的執行結果

執行「Run HGNN Model (KNeighbors).ipynb」的 Jupyter Notebook的檔案，需要有以下「./planetoid_dataset.py 、./utils/tsne.py 、./utils/norm.py 、./utils/convert.py」結構的檔案。



