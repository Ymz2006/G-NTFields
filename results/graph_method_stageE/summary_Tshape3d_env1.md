| model | seeds | cases | forward (%) | reverse (%) | either (%) | alternate (%) | collision | no-converge |
|---|---|---|---|---|---|---|---|---|
| metric NTFields baseline (native pipeline, reference) | 2 | 1000 | 81.2 ± 1.3 | 80.0 ± 0.1 | 91.1 ± 0.8 | 88.6 ± 0.4 | 117.5 ± 6.4 | 139.0 ± 14.1 |
| metric head (G-Shape), our pipeline | 3 | 1000 | 80.0 ± 0.6 | 80.1 ± 0.4 | 87.2 ± 0.2 | 82.1 ± 0.9 | 18.7 ± 6.0 | 188.3 ± 1.2 |
| graph_metric (graph latents, metric-type a) | 3 | 1000 | 79.2 ± 0.2 | 80.8 ± 0.1 | 87.3 ± 0.2 | 82.5 ± 0.7 | 20.0 ± 6.0 | 194.0 ± 3.6 |
| graph_gram (Euclidean a) | 3 | 1000 | 75.1 ± 0.4 | 76.2 ± 0.5 | 85.8 ± 0.3 | 80.0 ± 0.8 | 74.7 ± 10.0 | 200.3 ± 3.5 |
| graph_gram with k_a 128 | 1 | 1000 | 74.2 | 74.6 | 84.8 | 80.1 | 84.0 | 196.0 |
