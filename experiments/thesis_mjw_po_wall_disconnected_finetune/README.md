# PO-wall disconnected-start fine-tuning

This experiment continues a trained PO-wall TMASAC baseline for 20M additional
environment transitions. Each episode starts from a compact pre-connected layout
with all connection edges removed; normal reset settling is still applied. Training
uses the original 50-layout pool (seeds 42000--42049), while the baked-in frozen
evaluation hook uses a disjoint 50-layout pool (seeds 3000000--3000049).

Run one continuation for each source checkpoint:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_tmasac.py <checkpoint.pt>
```

The hook evaluates after +10M and +20M transitions (50% and 100% of the
fine-tuning window) and writes `eval_log.csv` in the continuation run directory.
Evaluation and live video recording are disabled so headless cluster runs do not
require an EGL display; this does not disable numerical evaluation.

Separate 50M launchers evaluate after +25M and +50M transitions and write under
`runs/thesis_mjw_po_wall_disconnected_finetune_50m/`:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_tmasac_50m.py <checkpoint.pt>
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_tmasac_no_connectors_50m.py <checkpoint.pt>
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_mat_qcx_50m.py <checkpoint.pt>
```

The no-connectors TMASAC variant exposes only actuator actions to the policy and
always sends `-1` for every physical connector. It restores the complete actuator
head and removes only the connector columns from critic/NOP action-input weights.
The old connector head is discarded and optimizer state starts fresh.

The MAT-QCX launcher expects the standard thesis MAT-QCX configuration: PPO with
1024 environments, four rollout steps per environment, sign-magnitude Beta actions,
and NOP enabled.

Generate grouped and per-run training curves for all three 50M variants with:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\plot_results.py
```

The plots are written to `experiments/thesis_mjw_po_wall_disconnected_finetune/results/`.
Record the normal and actuator-only TMASAC checkpoints respectively with:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\record_tmasac_50m.py <checkpoint.pt>
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\record_tmasac_no_connectors_50m.py <checkpoint.pt>
```

Both recorders default to the disjoint evaluation seed range and record 4- and
5-unit pools. Options after the checkpoint are forwarded to the shared recorder;
for example, append `--episodes 5 --cuda-idx 1`.

The zero-fine-tuning baseline is the existing disconnected-pool evaluation:

```powershell
.venv\Scripts\python.exe experiments\evaluate_thesis_mjw_disconnected_pools.py --target po_wall_tmasac --unit-counts 4 5
```

For the connection-use ablation, evaluate a fine-tuned checkpoint with connector
actions forced to disconnect:

```powershell
.venv\Scripts\python.exe experiments\evaluate_thesis_mjw_disconnected_pools_no_connections.py --target po_wall_tmasac --unit-counts 4 5 --po-wall-checkpoint <checkpoint.pt>
```
