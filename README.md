Easy interface for prototyping and running graph-related machine learning experiments

## Graph kernels

- [Graphlet sampling](./model_configs/graph_classification/mutag_graphlet_sampling.yml), [shortest path](./model_configs/graph_classification/mutag_shortest_path.yml) (implemented in [grakel](https://github.com/ysig/GraKeL))
- [Weisfeiler-Lehman](./model_configs/graph_classification/mutag_wl.yml)
- [Persistent Weisfeiler-Lehman](./model_configs/graph_classification/persistent_wl_subtree) ([paper](http://proceedings.mlr.press/v97/rieck19a.html) - [code](https://github.com/BorgwardtLab/P-WL))


## Node embedding

- [node2vec](./model_configs/node_classification/node2vec) ([paper](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) - [code](https://github.com/aditya-grover/node2vec))

## Others

- [Deep Learning with Topological Signatures](./model_configs/graph_classification/reddit5k_pd_vertex_degree.yml) ([paper](http://papers.nips.cc/paper/6761-deep-learning-with-topological-signatures) - [code](https://github.com/c-hofer/nips2017))

## Run experiments

This library [dlex](https://github.com/trungd/dlex) is required for running experiments.

```yaml
pip install dlex
python -m dlex.sklearn.train -c <path_to_yml_config>
```

## List of data sets

### Single graph

- [Karate Club](./src/datasets/karate_club.py)
- [BlogCatalog](./src/datasets/blog_catalog.py)

### Multiple graphs

- [Many benchmark data sets for graph kernels](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) (KKMMN2016 - loaded by [grakel](https://ysig.github.io/GraKeL/dev/generated/grakel.datasets.fetch_dataset.html#grakel.datasets.fetch_dataset))