# HyperWave-X V1 experiments

This directory contains the full method, six-scenario and two-probe launchers,
the two-scenario ablation, the wave-parameter sensitivity study, and the
publication plotting code. Run commands from this directory. All paths are
relative and the code does not require a particular environment name.

## Frozen protocol and result locations

- Main scenarios: `hetero_mixed`, `hetero_strong`, `consensus_wide`,
  `fog_sight1_level`, `dynamic_overload`, `scale_20`.
- Reviewer probes: `probe_complementary`, `probe_interaction`.
- Training seeds: `20 23 56 89 2025`.
- Main results: `results/FORMAL_6SCENARIOS_ORGANIZED_V1`.
- Probe results: `results/PROBES_ORGANIZED_V1`.
- Ablation results: `results/ablation_results`.
- Sensitivity results: `results/sensitivity-results`.
- Figure/table generator and outputs: `paper_artifacts`.

## Six main scenarios

```bash
python -u run_hyperwavex_formal.py --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --results-root ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --resume
```

## Reviewer probes

```bash
python -u run_hyperwavex_formal.py --include-probes --scenarios probe_complementary --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --results-root ./results/PROBES_ORGANIZED_V1 --resume
python -u run_hyperwavex_formal.py --include-probes --scenarios probe_interaction --train-seeds 104729 130363 155921 181081 205019 --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 31001 31002 31003 31004 31005 31006 31007 31008 31009 31010 --eval-episodes 20 --results-root ./results/PROBES_ORGANIZED_V1 --resume
```

## Ablation: two scenarios x three variants x five seeds

```bash
python -u run_hyperwavex_ablation.py --scenarios dynamic_overload scale_20 --variants no_capability no_wave complete_topology --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --full-results-roots ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --results-root ./results/ablation_results --resume
```

## Sensitivity: wave iterations and damping on consensus_wide

```bash
python -u run_hyperwavex_sensitivity.py --parameters wave_iterations wave_damping --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-episodes 20 --auto-discover-default-results --default-results-search-root ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --results-root ./results/sensitivity-results --resume
```

## Rebuild publication figures and tables

```bash
python -u paper_artifacts/plot_all_en.py
python -u paper_artifacts/plot_all_zh.py
```

Add `--dry-run` to training launchers to inspect schedules without training.
Paper-facing CSV files contain measured quantities and experiment identifiers;
checkpoint and execution provenance is stored in JSON protocol files.

## Probe 2 comparison

```bash
python compare_probe2.py --hypercomm-root ../HyperComm/results/PROBES_ORGANIZED_V1 --hyperwavex-root ./results/PROBES_ORGANIZED_V1 --no-wave-root ./results/PROBES_ORGANIZED_V1
```

See `paper_artifacts/FIGURE.TXT` for bilingual figure commands.

## Reviewer 2: cross-configuration generalization

This evaluation reuses the five trained `hetero_strong` checkpoints from
HyperWave-X, HyperComm, CommNet, IC3Net, MAGIC, QMIX, TarMAC, TeamComm, and
TieComm. It performs no training and compares the source configuration with
registered demand-shift and fleet-shift configurations. The launcher runs in
the HyperWave-X environment and invokes the TeamComm environment through the
explicit `--teamcomm-python` executable.

```bash
TEAMCOMM_PYTHON="$(conda run -n teamcomm python -c 'import sys; print(sys.executable)')" && python -u run_cross_configuration_generalization.py --methods hyperwave_x hypercomm commnet ic3net magic qmix tarmac teamcomm tiecomm --configurations source_configuration demand_shift fleet_shift --gpu -1 --workers 5 --cpu-threads-per-job 2 --teamcomm-python "$TEAMCOMM_PYTHON" --hyperwavex-source-root ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --hypercomm-source-root ../HyperComm/results/FORMAL_6SCENARIOS --teamcomm-source-root ../TeamComm/results --hyperwavex-results-root ./results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES --hypercomm-results-root ../HyperComm/results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES --teamcomm-results-root ../TeamComm/results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES --resume
```

The launcher resolves the TeamComm environment's Python executable through
Conda, runs 135 checkpoint-evaluation jobs (nine methods, three target
configurations, and five training seeds), and writes paper-ready summaries under
`results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES/paper_summary`.
