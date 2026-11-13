# Finite POMDP examples

Four finite POMDPs used in the paper:

| Example | Source | Notes |
|---------|--------|-------|
| Tiger | Kaelbling, Littman & Cassandra (1998) | `pomdps/tiger.95.POMDP` |
| Network | Cassandra `network.POMDP` (due to Littman, 1996) | `pomdps/network.POMDP` |
| Shopping-5 | Variant of [gym-pomdps](https://github.com/abaisero/gym-pomdps) Shopping-5 | Built in `shopping5.py` (see root README for differences) |
| Bridge patrol | This repository | Custom; 7 rows × 11 columns |

## Quick run

From the repository root:

```bash
python experiments/finite_pomdps/run.py --problem all --policy A
python experiments/finite_pomdps/run_bridge_patrol.py
python experiments/finite_pomdps/run_martingale.py
```

Paper figures and tables use the hand-policy folders (`*_hand`) plus Bridge patrol.

Or use the top-level entry point:

```bash
python reproduce.py finite
```

Artifacts are written under `data/runs/finite_pomdps/`.

## Bridge patrol checklist

- Grid: **7 rows × 11 columns**, 57 accessible cells + fall state `F`
- Shortest risky path 8, shortest safe (no bridge) path 14
- Tie-break order: down, right, up, left
- Paper CIs use `δ=0.025` per tail
