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
The zero-fine-tuning baseline is the existing disconnected-pool evaluation:

```powershell
.venv\Scripts\python.exe experiments\evaluate_thesis_mjw_disconnected_pools.py --target po_wall_tmasac --unit-counts 4 5
```

For the connection-use ablation, evaluate a fine-tuned checkpoint with connector
actions forced to disconnect:

```powershell
.venv\Scripts\python.exe experiments\evaluate_thesis_mjw_disconnected_pools_no_connections.py --target po_wall_tmasac --unit-counts 4 5 --po-wall-checkpoint <checkpoint.pt>
```
