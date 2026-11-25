# CAHC V1 experiments

This directory contains the full method, six-scenario and two-probe launchers,
the two-scenario ablation, the wave-parameter sensitivity study, and the
publication plotting code. Run commands from this directory. All paths are
relative and the code does not require a particular environment name.

## Frozen protocol and result locations

- Main scenarios: `hetero_mixed`, `hetero_strong`, `consensus_wide`,
  `fog_sight1_level`, `dynamic_overload`, `scale_20`.
- Controlled probes: `probe_complementary`, `probe_interaction`.
- Training seeds: `20 23 56 89 2025`.
- Main results: `results/FORMAL_6SCENARIOS_ORGANIZED_V1`.
- Probe results: `results/PROBES_ORGANIZED_V1`.
- Ablation results: `results/ablation_results`.
- Sensitivity results: `results/sensitivity-results`.
- Graph-refinement controls: `results/graph_intelligence/refinement_controls`.
- Message-dropout robustness: `results/graph_intelligence/message_dropout`.
- Figure/table generator and outputs: `paper_artifacts`.

## Six main scenarios

```bash
python -u run_cahc_formal.py --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --results-root ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --resume
```

## Controlled probes

```bash
python -u run_cahc_formal.py --include-probes --scenarios probe_complementary --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --results-root ./results/PROBES_ORGANIZED_V1 --resume
python -u run_cahc_formal.py --include-probes --scenarios probe_interaction --train-seeds 104729 130363 155921 181081 205019 --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 31001 31002 31003 31004 31005 31006 31007 31008 31009 31010 --eval-episodes 20 --results-root ./results/PROBES_ORGANIZED_V1 --resume
```

## Ablation: two scenarios x three variants x five seeds

```bash
python -u run_cahc_ablation.py --scenarios dynamic_overload scale_20 --variants no_capability no_wave complete_topology --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --full-results-roots ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --results-root ./results/ablation_results --resume
```

## Sensitivity: wave iterations and damping on consensus_wide

```bash
python -u run_cahc_sensitivity.py --parameters wave_iterations wave_damping --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-episodes 20 --auto-discover-default-results --default-results-search-root ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --results-root ./results/sensitivity-results --resume
```

## Rebuild publication figures and tables

```bash
python -u paper_artifacts/plot_all_en.py
python -u paper_artifacts/plot_all_zh.py
```

The English figures are written to `paper_artifacts/figures_english`, and the
Chinese figures are written to `paper_artifacts/figures_chinese`. Both commands
read existing result files and do not train or reevaluate policies. Derived
tables are written to `paper_artifacts/tables_en` and
`paper_artifacts/tables_zh`.

Add `--dry-run` to training launchers to inspect schedules without training.
Paper-facing CSV files contain measured quantities and experiment identifiers;
checkpoint and execution provenance is stored in JSON protocol files.

## Probe 2 comparison

```bash
python compare_probe2.py --hypercomm-root ../HyperComm/results/PROBES_ORGANIZED_V1 --cahc-root ./results/PROBES_ORGANIZED_V1 --no-wave-root ./results/PROBES_ORGANIZED_V1
```

## Graph Intelligence: controlled refinement operators

This experiment holds the task-conditioned hypergraph constructor, learned
execution topology, policy/value backbone, training budget, and seeds fixed.
It compares full input-anchored refinement with no refinement, ordinary
normalized hypergraph diffusion, and repeated two-head GAT propagation on
`fog_sight1_level` and `scale_20`. The ten full-method checkpoints are reused
and freshly evaluated with the common evaluator; the three controls require
30 new training jobs.

Run the launcher from the `CAHC` directory. Project inputs and outputs
must be supplied as relative paths; absolute paths and paths escaping this
directory are rejected. The launcher uses the active Python interpreter for
all child jobs, so the same command works after activating the project
environment in a PyCharm terminal.

```bash
python -u run_graph_refinement_controls.py --scenarios fog_sight1_level scale_20 --variants no_wave plain_diffusion repeated_gat --gpu -1 --workers 5 --cpu-threads-per-job 2 --full-results-root ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --results-root ./results/graph_intelligence/refinement_controls --resume
```

Windows PyCharm terminal (PowerShell, CPU, five seeds in parallel):

```powershell
if (!(Test-Path ".\run_graph_refinement_controls.py")) { throw "Run this command from the CAHC directory." }; if (!(Test-Path ".\results\FORMAL_6SCENARIOS_ORGANIZED_V1")) { throw "Missing the formal CAHC checkpoints under the relative results path." }; $env:PYTHONUNBUFFERED="1"; $env:PYTHONUTF8="1"; $env:CUDA_VISIBLE_DEVICES=""; $env:OMP_NUM_THREADS="2"; $env:MKL_NUM_THREADS="2"; $env:OPENBLAS_NUM_THREADS="2"; $env:NUMEXPR_NUM_THREADS="2"; python -u .\run_graph_refinement_controls.py --scenarios fog_sight1_level scale_20 --variants no_wave plain_diffusion repeated_gat --gpu -1 --workers 5 --cpu-threads-per-job 2 --full-results-root ".\results\FORMAL_6SCENARIOS_ORGANIZED_V1" --results-root ".\results\graph_intelligence\refinement_controls" --resume; if ($LASTEXITCODE -ne 0) { throw "Graph-refinement controls failed with exit code $LASTEXITCODE" }
```

The two scenarios may also run on separate Windows machines. Single-scenario
runs use scenario-qualified manifest and summary names, so their result trees
can later be merged without overwriting metadata. Keep the same relative
`results-root` on both machines and copy the completed tree from one machine
into the other with directory merging enabled.

Windows machine 1 (`fog_sight1_level`):

```powershell
if (!(Test-Path ".\run_graph_refinement_controls.py")) { throw "Run this command from the CAHC directory." }; if (!(Test-Path ".\results\FORMAL_6SCENARIOS_ORGANIZED_V1")) { throw "Missing the formal CAHC checkpoints." }; $env:PYTHONUNBUFFERED="1"; $env:PYTHONUTF8="1"; $env:CUDA_VISIBLE_DEVICES=""; $env:OMP_NUM_THREADS="2"; $env:MKL_NUM_THREADS="2"; $env:OPENBLAS_NUM_THREADS="2"; $env:NUMEXPR_NUM_THREADS="2"; python -u .\run_graph_refinement_controls.py --scenarios fog_sight1_level --variants no_wave plain_diffusion repeated_gat --gpu -1 --workers 5 --cpu-threads-per-job 2 --full-results-root ".\results\FORMAL_6SCENARIOS_ORGANIZED_V1" --results-root ".\results\graph_intelligence\refinement_controls" --resume; if ($LASTEXITCODE -ne 0) { throw "fog_sight1_level controls failed with exit code $LASTEXITCODE" }
```

Windows machine 2 (`scale_20`):

```powershell
if (!(Test-Path ".\run_graph_refinement_controls.py")) { throw "Run this command from the CAHC directory." }; if (!(Test-Path ".\results\FORMAL_6SCENARIOS_ORGANIZED_V1")) { throw "Missing the formal CAHC checkpoints." }; $env:PYTHONUNBUFFERED="1"; $env:PYTHONUTF8="1"; $env:CUDA_VISIBLE_DEVICES=""; $env:OMP_NUM_THREADS="2"; $env:MKL_NUM_THREADS="2"; $env:OPENBLAS_NUM_THREADS="2"; $env:NUMEXPR_NUM_THREADS="2"; python -u .\run_graph_refinement_controls.py --scenarios scale_20 --variants no_wave plain_diffusion repeated_gat --gpu -1 --workers 5 --cpu-threads-per-job 2 --full-results-root ".\results\FORMAL_6SCENARIOS_ORGANIZED_V1" --results-root ".\results\graph_intelligence\refinement_controls" --resume; if ($LASTEXITCODE -ne 0) { throw "scale_20 controls failed with exit code $LASTEXITCODE" }
```

After merging both completed result trees onto one machine, rerun the two-scenario
command above with `--resume`. It validates all 40 jobs and creates the combined
two-scenario summary without retraining completed jobs.

## Graph Intelligence: zero-shot message-dropout robustness

This experiment performs no training. It reuses the ten full checkpoints and
applies deterministic, nested dropout masks to decoded directed links before
GAT message aggregation. The 0%, 10%, 20%, and 30% conditions use the same
evaluation cases and do not perturb the environment or policy RNG streams.
The launcher must run in the CAHC Python environment; it checks for
PyTorch before dispatching any evaluation jobs.

```bash
conda run --no-capture-output -n hypercomm39 python -u run_message_dropout_robustness.py --scenarios fog_sight1_level scale_20 --dropout-rates 0 0.1 0.2 0.3 --gpu -1 --workers 5 --cpu-threads-per-job 2 --source-results-root ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --results-root ./results/graph_intelligence/message_dropout --resume
```

Windows PyCharm terminal (PowerShell, frozen checkpoints, five evaluations in
parallel):

```powershell
if (!(Test-Path ".\run_message_dropout_robustness.py")) { throw "Run this command from the CAHC directory." }; if (!(Test-Path ".\results\FORMAL_6SCENARIOS_ORGANIZED_V1")) { throw "Missing the formal CAHC checkpoints under the relative results path." }; $env:PYTHONUNBUFFERED="1"; $env:PYTHONUTF8="1"; $env:CUDA_VISIBLE_DEVICES=""; $env:OMP_NUM_THREADS="2"; $env:MKL_NUM_THREADS="2"; $env:OPENBLAS_NUM_THREADS="2"; $env:NUMEXPR_NUM_THREADS="2"; python -u .\run_message_dropout_robustness.py --scenarios fog_sight1_level scale_20 --dropout-rates 0 0.1 0.2 0.3 --gpu -1 --workers 5 --cpu-threads-per-job 2 --source-results-root ".\results\FORMAL_6SCENARIOS_ORGANIZED_V1" --results-root ".\results\graph_intelligence\message_dropout" --resume; if ($LASTEXITCODE -ne 0) { throw "Message-dropout evaluation failed with exit code $LASTEXITCODE" }
```

## Cross-configuration generalization

This evaluation reuses the five trained `hetero_strong` checkpoints from
CAHC, HyperComm, CommNet, IC3Net, MAGIC, QMIX, TarMAC, TeamComm, and
TieComm. It performs no training and compares the source configuration with
registered demand-shift and fleet-shift configurations. The launcher runs in
the CAHC environment and invokes the TeamComm environment through the
explicit `--teamcomm-python` executable.

```bash
TEAMCOMM_PYTHON="$(conda run -n teamcomm python -c 'import sys; print(sys.executable)')" && python -u run_cross_configuration_generalization.py --methods cahc hypercomm commnet ic3net magic qmix tarmac teamcomm tiecomm --configurations source_configuration demand_shift fleet_shift --gpu -1 --workers 5 --cpu-threads-per-job 2 --teamcomm-python "$TEAMCOMM_PYTHON" --cahc-source-root ./results/FORMAL_6SCENARIOS_ORGANIZED_V1 --hypercomm-source-root ../HyperComm/results/FORMAL_6SCENARIOS --teamcomm-source-root ../TeamComm/results --cahc-results-root ./results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES --hypercomm-results-root ../HyperComm/results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES --teamcomm-results-root ../TeamComm/results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES --resume
```

The launcher resolves the TeamComm environment's Python executable through
Conda, runs 135 checkpoint-evaluation jobs (nine methods, three target
configurations, and five training seeds), and writes paper-ready summaries under
`results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES/paper_summary`.
