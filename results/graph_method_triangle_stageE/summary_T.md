| model | seeds | cases | forward (%) | reverse (%) | either (%) | alternate (%) | collision | no-converge |
|---|---|---|---|---|---|---|---|---|
| metric head (graph_method) | 3 | 1000 | 80.0 ± 0.6 | 80.1 ± 0.4 | 87.2 ± 0.2 | 82.1 ± 0.9 | 18.7 ± 6.0 | 188.3 ± 1.2 |
| graph_metric (graph_method) | 3 | 1000 | 79.2 ± 0.2 | 80.8 ± 0.1 | 87.3 ± 0.2 | 82.5 ± 0.7 | 20.0 ± 6.0 | 194.0 ± 3.6 |
| graph_gram (graph_method) | 3 | 1000 | 75.1 ± 0.4 | 76.2 ± 0.5 | 85.8 ± 0.3 | 80.0 ± 0.8 | 74.7 ± 10.0 | 200.3 ± 3.5 |
| graph_tri (Euclidean a, free V) | 1 | 1000 | 76.1 | 75.5 | 85.4 | 80.1 | 56.0 | 201.0 |
| graph_tri_metric, free V, random init | 1 | 1000 | 72.9 | 74.8 | 86.0 | 76.4 | 89.0 | 197.0 |
| graph_tri_metric, free V | 3 | 1000 | 75.6 ± 1.8 | 77.4 ± 1.0 | 86.9 ± 1.3 | 80.9 ± 2.3 | 60.0 ± 18.7 | 202.0 ± 5.6 |
| **graph_tri_metric, equivariant V** | 3 | 1000 | 77.2 ± 2.2 | 77.7 ± 2.4 | 85.3 ± 1.5 | 80.4 ± 2.4 | 73.7 ± 23.5 | 171.3 ± 12.6 |
| graph_tri_metric free V, l_1 | 1 | 1000 | 74.0 | 77.4 | 87.0 | 83.6 | 67.0 | 211.0 |
| graph_tri_metric free V, l_2 | 1 | 1000 | 72.2 | 73.0 | 84.8 | 78.3 | 107.0 | 204.0 |
| graph_tri_metric free V, l_4 | 1 | 1000 | 72.2 | 74.9 | 85.5 | 75.3 | 98.0 | 204.0 |
| graph_tri_metric free V, l_16 | 1 | 1000 | 73.0 | 76.1 | 86.4 | 78.1 | 85.0 | 204.0 |
| graph_tri_metric free V, l_32 | 1 | 1000 | 73.3 | 76.5 | 86.0 | 80.2 | 85.0 | 206.0 |
