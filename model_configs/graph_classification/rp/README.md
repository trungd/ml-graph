# Training

```
python -m dlex.train -c model_configs/graph_classification/rp/pd_multi_rpf_weighted_deep_sets.yml --report --show-progress --env <dataset> --g <gpu>
```

where 
- `-g`: index of gpu(s) that will be used for training
- `--env`: name of the environment used for training

Examples

```
python -m dlex.train -c model_configs/graph_classification/rp/pd_multi_rpf_weighted_deep_sets.yml --report --show-progress --env reddit5k --g 0
```
