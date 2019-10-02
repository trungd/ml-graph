Easy interface for prototyping and running graph-related machine learning experiments

## Graph kernels

- [Graphlet sampling](./model_configs/graph_classification/mutag_graphlet_sampling.yml)
- [Weisfeiler-Lehman](./model_configs/graph_classification/mutag_wl.yml)
- [Persistent Weisfeiler-Lehman](./model_configs/graph_classification/persistent_wl_subtree) ([paper](http://proceedings.mlr.press/v97/rieck19a.html) - [code](https://github.com/BorgwardtLab/P-WL))


## Node embedding

- [node2vec](./model_configs/node_classification/node2vec)

## Run experiments

This library [dlex](https://github.com/trungd/dlex) is required for running experiments.

```yaml
pip install dlex
python -m dlex.sklearn.train -c <path_to_yml_config>
```