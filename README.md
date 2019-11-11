Easy interface for prototyping and running graph-related machine learning experiments. The code uses scikit-learn or pytorch depending on each implementation.

## Graph kernels

- [Graphlet sampling](./model_configs/graph_classification/mutag_graphlet_sampling.yml), [shortest path](./model_configs/graph_classification/mutag_shortest_path.yml) (implemented in [grakel](https://github.com/ysig/GraKeL))
- [Weisfeiler-Lehman](./model_configs/graph_classification/mutag_wl.yml)
- [Persistent Weisfeiler-Lehman](./model_configs/graph_classification/persistent_wl_subtree) ([paper](http://proceedings.mlr.press/v97/rieck19a.html) - [code](https://github.com/BorgwardtLab/P-WL))


## Node embedding

- [node2vec](./model_configs/node_classification/node2vec) ([paper](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) - [code](https://github.com/aditya-grover/node2vec))

## Others

- [Deep Learning with Topological Signatures](./model_configs/graph_classification/reddit5k_pd_vertex_degree.yml) ([paper](http://papers.nips.cc/paper/6761-deep-learning-with-topological-signatures) - [code](https://github.com/c-hofer/nips2017))
- [PersLay]() - ([paper]() - [code](https://github.com/MathieuCarriere/perslay))

## Run experiments

This library [dlex](https://github.com/trungd/dlex) is required for running experiments.

```yaml
pip install dlex
python -m dlex.train -c <path_to_yml_config>
```

## List of data sets

### Single graph

- [Karate Club](./src/datasets/karate_club.py)
- [BlogCatalog](./src/datasets/blog_catalog.py)

### Multiple graphs

- [Many benchmark data sets for graph kernels](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) (KKMMN2016 - loaded by [grakel](https://ysig.github.io/GraKeL/dev/generated/grakel.datasets.fetch_dataset.html#grakel.datasets.fetch_dataset))

## Results

### Graph Classification

Results may not be comparable to original papers (due to different settings)

| Model | MUTAG | NCI109 | REDDIT-5K | REDDIT-12K | COLLAB |
|-------|-------|--------|-----------|------------|--------|
|PWL (H0) [Rieck, 2019] | 82.46 | 81.66 |  
|PWLC (H0 + H1) [Rieck, 2019] | 83.54 | 82.46 | 
|DLTopo (H0 non-essential pairs) [Hofer, 2017] | | | 43.0 |
|DLTopo (H0 + H1) [Hofer, 2017] | | | 54.4 | 46.4 |
|Perslay [Carriere, 2019] | | | |