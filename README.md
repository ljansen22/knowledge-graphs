# Healthcare Fraud Detection via Knowledge Graphs

A workshop project for exploring open-source Python knowledge graph tools to detect fraud in healthcare claims data.

## What's in here

```
data/
  generate_dataset.py      — synthetic data generator (run this first)
  patients.csv             — 300 patients (4 deceased)
  providers.csv            — 60 providers (5 fraudulent)
  claims.csv               — all claims WITH fraud labels  ← instructor only
  claims_public.csv        — all claims WITHOUT labels     ← distribute this
  referrals.csv            — provider-to-provider referrals

notebooks/
  01_solution_fraud_detection.ipynb  — full solution with all 5 patterns  ← instructor only
  02_workshop_challenge.ipynb        — competition notebook for participants
```

## Five fraud patterns embedded in the data

| # | Pattern | Fraudster | Signal |
|---|---------|-----------|--------|
| 1 | Ghost Billing | PRV-047 | Claims filed after patient date_of_death |
| 2 | Referral Ring | PRV-031/032/033 | Closed Louvain community, inflated amounts (2–3×) |
| 3 | Impossible Travel | PAT-0089 | Same-day claims in cities 1,300 miles apart |
| 4 | Upcoding | PRV-022 | 99215 billing rate 8× above peers (3σ outlier) |
| 5 | Duplicate Billing | PRV-055 | Parallel edges in multigraph (same patient/procedure/date) |

## Quick start

```bash
# Install dependencies
uv sync

# Generate the dataset
uv run python data/generate_dataset.py

# Launch Jupyter
uv run jupyter notebook
```

Then open `notebooks/01_solution_fraud_detection.ipynb` (instructor) or
`notebooks/02_workshop_challenge.ipynb` (participants).

## Key packages

| Package | Role |
|---------|------|
| `networkx` | Graph construction, traversal, subgraph analysis |
| `python-louvain` | Louvain community detection (`import community`) |
| `pyvis` | Interactive HTML graph visualisation |
| `pandas` / `numpy` | Data wrangling |
| `matplotlib` / `seaborn` | Static plots |

## Workshop format

1. Distribute `data/claims_public.csv`, `data/patients.csv`, `data/providers.csv`, `data/referrals.csv`
2. Each team works through `notebooks/02_workshop_challenge.ipynb`
3. Teams submit findings by calling `score_team('Team Name')` — max 100 pts
4. Run `notebooks/01_solution_fraud_detection.ipynb` for the debrief
