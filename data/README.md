# Experiment data layout

```
data/
  runs/
    finite_pomdps/           # Tiger, Network, Shopping-5, Bridge patrol
    continuous_baselines/    # five LQG examples
    ablation_poly_degree/    # poly-degree ablation summaries / tables
    paper_final/             # aggregated paper numbers from collect_paper_final.py
    paper_meta.json          # computing environment + paper metadata
  catalog/
    index.json               # optional index of run summaries
```

## What is committed vs regenerated

| Artifact | In git? | Purpose |
|----------|---------|---------|
| `.POMDP` sources, LQG JSON configs | yes | model definitions |
| small `summary.json` / `config.json` | optional | convenience snapshots |
| `*.npz`, PNGs, PDFs | **no** (gitignored) | regenerate with `python reproduce.py …` |
| `results/paper_numbers.json` | yes (after `tables`) | compact paper numbers |

Refresh the catalog after adding runs:

```bash
python scripts/build_catalog.py
```
