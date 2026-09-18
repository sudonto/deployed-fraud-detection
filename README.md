# Deployed fraud detection: the label feedback loop

Simulation code for the paper *The label feedback loop in deployed fraud
detection and how to close it*. A deployed detector decides which transactions
get investigated, and the investigations become its next training set;
everything unreviewed is filed as legitimate. The study replays three public
transaction streams through that loop and compares bookkeeping rules.

Every number in the manuscript is generated from the raw run files. Nothing is
transcribed by hand.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # Unix
pip install -r requirements.txt
```

Download the Paper 1 streams from OpenML and check the published shapes:

```bash
python -m scripts.fetch_datasets baf_base baf_variant3 ieee_cis ulb_creditcard
python -m scripts.verify_modules
```

## Running the study

```bash
# Structural checks on the review policies and the bookkeeping rules
python -m scripts.smoke_feedback

# Headline study: the loop, its capacity sweep, the detector sensitivity
# sweep, and the robustness stream
python -m src.paper1_fraud.run_feedback --stage all

# Negative-result track: baselines, resampling, encoder ablation, leakage
python -m src.paper1_fraud.run_experiments --stage all
```

Both runners skip an arm whose JSON already exists. Pass `--force` to recompute.

```bash
python -m src.analysis.run_analysis --paper 1
```

regenerates the tables, figures and LaTeX macros from the raw run files.

## License

MIT. See `LICENSE`.
