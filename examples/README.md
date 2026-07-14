# Worked examples

Each script is self-contained and runs in a few seconds on CPU with no downloads.

```bash
pip install -e ".[examples]"
python examples/01_screening_classifier.py
python examples/02_group_rate_gap.py
```

- **01_screening_classifier.py** — a logistic-regression flag rate under all four
  strata. The labeling and sampling strata dominate; the honest band comes out
  about nine times wider than the bootstrap alone. Verdict: the model-only error
  bar is cosmetic.

- **02_group_rate_gap.py** — a difference in positive-outcome rates between two
  groups, where one group is over-sampled. The outer strata are present but do
  not dominate, and the framework correctly reports that the statistical band is
  a reasonable summary. This is the counter-case: the point is honest widening,
  not maximal widening.
