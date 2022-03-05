# Clustering Code

## Clustering 

Run the following command:

```
python cluster_eval.py data.npz
```

where data.npz contains "labels" and "features"

Hierarchical agglomerative clustering will be run and F1, recall, precision, NMI, and clustering accuracy will all be reported.

To create features from the available methods, visit the corresponding get\_feats\_{method\_name}.py and adjust the args as needed. Then run:

```
python get_feats_pdr.py # or any alternative to pdr
```
