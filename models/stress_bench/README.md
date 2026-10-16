# stress_bench: flaw-exposing planning benchmark of the travel-time fields (2026-10-13/14)

Seven fields (original metric NTFields baseline; graph_gram; graph_metric; graph_metric + escape-time potential;
cover euclid pseudometric; cover euclid + conformal cone; branch K=4 + conformal cone) planned with the official
forward MPPI (`models/graph_method_triangle/scripts/plan_eval_graph.py`, batch 125, seed 0, SE(3) criterion) on
14 sets per scene. Report: `results/stress_bench/REPORT.md` (findings + all tables), page
`results/stress_bench/report.html`, summary `results/stress_bench/summary/stress_summary.{md,json}`.

Pipeline (run order; everything in the pytorchserver Docker image except the host-side queue):

    make_stress_sets.py   # tightboth tightgoal tightrot blocked squeeze ctrl + reversed copies -> datasets/3dshape/bench
    run_stress.py         # host queue: docker containers, GPU-memory aware, reuses identical earlier results
    field_checks.py       # identity / symmetry / triangle inequality / detour factor per checkpoint
    (models/graph_lie/scripts/speed_range_eval.py for the near-contact speed calibration)
    aggregate_stress.py   # -> summary tables; make_report.py + findings.json -> REPORT.md / report.html

Headline: tight-space sets and reversed queries separate the heads (collisions 34-41 % for the baseline, graph_gram
and the pseudometric vs 4-6 % for branch + cone); every head predicts the near-contact speed 26-47x too fast.
