"""Simulation of the label feedback loop in a deployed fraud detector.

A fraud model in production does not get to see the labels of the transactions
it lets through. Analysts investigate the alerts the model raises, and those
investigations are the only source of ground truth. Whatever the model waves
past is recorded as legitimate by default, regardless of what it really was.
Retraining on that record closes a loop: fraud the model cannot see becomes
fraud the model is taught does not exist, and the blind spot is inherited by
every subsequent version of the model.

This module runs that loop forward on a real transaction stream so the damage
can be measured rather than argued about. The stream is replayed in rounds. In
each round the current model scores the incoming transactions, a review policy
spends a fixed analyst budget on some of them, the true labels of exactly those
transactions are revealed, and the model is retrained on whatever the arm's
bookkeeping rule puts in the pool. Because we hold the full ground truth for
the whole stream, every round can also be scored against labels the simulated
bank never had, which is what makes the degradation visible.

The arms differ only in the review policy and the bookkeeping rule, so any gap
between them is attributable to those two choices and nothing else.

Review policies
---------------
``topk``
    Spend the whole budget on the highest-scoring transactions. What a bank
    does by default, and the policy that closes the loop hardest.
``uniform``
    Hold back a fraction of the budget and spend it on transactions drawn
    uniformly at random. The textbook way to keep a policy exploring.
``banded``
    Hold back a fraction of the budget and spend it just below the alert
    cutoff, where the model is least certain. At a fraud rate of 0.17% a
    uniform draw is almost always a legitimate transaction, so the exploration
    budget buys far more information when it is aimed at the decision boundary.

Bookkeeping rules
-----------------
``assume_negative``
    Reviewed transactions carry their true label; everything else is recorded
    as legitimate. This is the rule that actually creates the feedback loop.
``reviewed_only``
    Only reviewed transactions enter the pool. No fabricated labels, but the
    pool is a heavily selected sample and its fraud rate is nothing like the
    population's.
``ips``
    Reviewed transactions enter the pool weighted by the reciprocal of their
    probability of having been reviewed. Under a stochastic policy with known
    propensities this makes the training loss an unbiased estimate of the loss
    the model would incur on the full stream.
``censor``
    Every transaction enters the pool as a bank would record it, except those
    the model scored highly and nobody reviewed, which are dropped instead of
    being recorded as legitimate. The missed frauds are not spread evenly
    through the unreviewed rows: they sit just under the alert cutoff, in the
    band the model already finds suspicious. Recording that band as legitimate
    is what teaches the next model to stop finding it suspicious. Dropping it
    costs some true negatives, which are the one thing this problem has in
    abundance, and requires no extra analyst capacity, which is the one thing
    it does not.
``censor_ips``
    Censoring plus inverse propensity weights on the reviewed rows.
``soft_censor``
    The same idea without a cutoff. Every unreviewed transaction is recorded as
    legitimate, as a bank would, but carries a weight equal to the model's own
    estimate that it really is legitimate. A row the model already suspects
    contributes almost nothing to the negative class, so it cannot teach the
    next model that its own suspicion was wrong, while a row the model is
    confident about counts in full. This keeps every row and needs no cutoff to
    tune, and the estimate it relies on is one the detector already produces.
``ips_augmented``
    Every transaction enters the pool, unreviewed ones recorded as legitimate
    as a bank would, but reviewed ones carry their true label and a weight of
    one over their propensity. The reasoning is that the assumed negatives are
    not the problem: at a fraud rate of 0.17% almost all of them are correct,
    and there are enough of them to matter. The problem is that the handful
    which are wrong are exactly the frauds the model already fails on, and no
    amount of correct negatives compensates for that. Weighting each confirmed
    fraud by the number of transactions it stands for restores the mass the
    unreviewed frauds should have contributed, while keeping the sample size
    that inverse weighting on its own throws away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REVIEW_POLICIES = ("topk", "uniform", "banded")
POOL_RULES = (
    "full_labels", "assume_negative", "reviewed_only",
    "ips", "ips_augmented", "censor", "censor_ips", "soft_censor",
)

# Rules that need a propensity bounded away from zero to weight by.
WEIGHTED_RULES = ("ips", "ips_augmented", "censor_ips")

# Rules that drop the unreviewed rows the model scored highly.
CENSORING_RULES = ("censor", "censor_ips")


@dataclass
class FeedbackConfig:
    """One arm of the simulation."""

    review_policy: str = "topk"
    pool_rule: str = "assume_negative"

    # Share of the review budget held back from the top of the ranking and spent
    # on exploration instead. Ignored by the ``topk`` policy.
    exploration: float = 0.0

    # Width of the uncertainty band, as a multiple of the budget. The band sits
    # immediately below the alerts the budget already covers, so a width of 10
    # with a budget of 200 draws from ranks 200 to 2200.
    band_width: float = 10.0

    # Floor on the review probability of any transaction. Inverse propensity
    # weights are unbounded as the propensity goes to zero, and a policy that
    # can never review a region gives no information about it at any weight. A
    # small uniform floor keeps every propensity positive and every weight
    # finite, at the cost of a few reviews per round.
    propensity_floor: float = 0.02

    # Weights above this multiple of the mean are clipped. Even with a floor,
    # the rarest propensities produce weights large enough that a single row
    # dominates the gradient; clipping trades a little bias for a variance that
    # does not swamp the correction.
    weight_clip: float = 20.0

    # How far below the alert cutoff the censored region reaches, as a multiple
    # of the review budget. Censoring one budget's worth drops only the rows
    # that just missed being reviewed; censoring twenty reaches most of the way
    # into the tail where the missed frauds actually are, at the price of
    # discarding more genuine negatives.
    censor_width: float = 10.0

    # Smallest weight a soft-censored row can carry.
    soft_floor: float = 0.01

    # Overrides merged over DETECTOR_PARAMS. The detector is held fixed across
    # every arm of the main study; this field exists so a sensitivity sweep can
    # vary one hyperparameter at a time without touching the module constant.
    detector: dict | None = None

    def label(self) -> str:
        parts = [self.review_policy, self.pool_rule]
        if self.review_policy != "topk":
            parts.append(f"eps{self.exploration:g}")
        if self.pool_rule in CENSORING_RULES:
            parts.append(f"c{self.censor_width:g}")
        return "+".join(parts)


@dataclass
class RoundOutcome:
    """What happened in one round, from both the bank's and the oracle's view."""

    round_index: int
    n_transactions: int
    n_frauds: int
    n_reviewed: int

    # Frauds the analysts actually found this round. The operational bottom line.
    frauds_caught: int

    # Scored against the labels the simulated bank never saw.
    auprc: float
    roc_auc: float
    recall_at_budget: float
    precision_at_budget: float

    # State of the training pool the model was fitted on, which is what the
    # degradation story is about.
    pool_size: int
    pool_positives: int
    pool_mislabelled: int


def _propensities(
    scores: np.ndarray, budget: int, config: FeedbackConfig
) -> np.ndarray:
    """Probability that each transaction is selected for review.

    Written out explicitly rather than estimated, because the simulation is the
    logging policy and there is no reason to model something we control. The
    result is a genuine probability per transaction under sampling without
    replacement, approximated by the per-draw share, which is what the inverse
    propensity weights need.
    """
    n = len(scores)
    budget = min(budget, n)
    order = np.argsort(-scores, kind="stable")

    floor_share = config.propensity_floor if config.pool_rule in WEIGHTED_RULES else 0.0
    explore_share = config.exploration if config.review_policy != "topk" else 0.0
    # The floor is carved out of the exploration budget where one exists, so
    # that adding a floor does not silently shrink the number of alerts.
    explore_share = max(explore_share, floor_share)

    n_explore = int(round(budget * explore_share))
    n_floor = int(round(budget * floor_share))
    n_explore = max(n_explore, n_floor)
    n_top = budget - n_explore

    probability = np.zeros(n, dtype=float)
    probability[order[:n_top]] = 1.0

    remaining = n_explore - n_floor
    if remaining > 0:
        if config.review_policy == "banded":
            band_end = min(n, n_top + int(round(budget * config.band_width)))
            band = order[n_top:band_end]
        else:
            band = order[n_top:]
        if len(band) > 0:
            probability[band] += remaining / len(band)

    if n_floor > 0:
        pool = order[n_top:]
        if len(pool) > 0:
            probability[pool] += n_floor / len(pool)

    return np.clip(probability, 0.0, 1.0)


def _select_reviews(
    scores: np.ndarray, budget: int, config: FeedbackConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Choose which transactions the analysts investigate this round.

    Returns the selected positions and the propensity of every transaction, so
    the caller can weight the labels it receives.
    """
    n = len(scores)
    budget = min(budget, n)
    propensity = _propensities(scores, budget, config)

    order = np.argsort(-scores, kind="stable")
    certain = order[propensity[order] >= 1.0]
    selected = list(certain)

    stochastic = np.setdiff1d(np.arange(n), certain, assume_unique=False)
    weights = propensity[stochastic]
    n_draw = min(budget - len(selected), int((weights > 0).sum()))
    if n_draw > 0:
        probabilities = weights / weights.sum()
        drawn = rng.choice(stochastic, size=n_draw, replace=False, p=probabilities)
        selected.extend(drawn.tolist())

    return np.array(sorted(selected), dtype=int), propensity


@dataclass
class LabelPool:
    """The training set as the simulated bank knows it.

    Holds the features, whatever label the bank recorded, a per-row weight, and
    the true label. The true label is never given to the model; it is carried
    only so the analysis can report how much of the pool is wrong, which is the
    quantity the whole paper is about.
    """

    X: list = field(default_factory=list)
    y: list = field(default_factory=list)
    w: list = field(default_factory=list)
    y_true: list = field(default_factory=list)

    def add(self, X, y, w, y_true) -> None:
        self.X.append(np.asarray(X, dtype=np.float32))
        self.y.append(np.asarray(y, dtype=np.int8))
        self.w.append(np.asarray(w, dtype=np.float64))
        self.y_true.append(np.asarray(y_true, dtype=np.int8))

    def arrays(self):
        return (
            np.vstack(self.X),
            np.concatenate(self.y),
            np.concatenate(self.w),
            np.concatenate(self.y_true),
        )

    def stats(self) -> tuple[int, int, int]:
        y = np.concatenate(self.y)
        y_true = np.concatenate(self.y_true)
        return len(y), int(y.sum()), int((y != y_true).sum())


def censored_mask(
    scores: np.ndarray, reviewed: np.ndarray, budget: int, config: FeedbackConfig
) -> np.ndarray:
    """Which unreviewed rows are dropped rather than recorded as legitimate.

    The region starts at the alert cutoff and runs down the ranking for a
    multiple of the budget. Reviewed rows are never censored: their labels are
    the only ones we actually know.
    """
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    width = int(round(budget * (1.0 + config.censor_width)))
    censored = np.zeros(n, dtype=bool)
    censored[order[:min(width, n)]] = True
    censored[reviewed] = False
    return censored


def update_pool(
    pool: LabelPool,
    X: np.ndarray,
    y_true: np.ndarray,
    reviewed: np.ndarray,
    propensity: np.ndarray,
    config: FeedbackConfig,
    scores: np.ndarray | None = None,
    budget: int = 0,
) -> None:
    """Record the round in the pool according to the arm's bookkeeping rule."""
    rule = config.pool_rule

    if rule in CENSORING_RULES:
        censored = censored_mask(scores, reviewed, budget, config)
        keep = ~censored
        recorded = np.zeros(len(y_true), dtype=np.int8)
        recorded[reviewed] = y_true[reviewed]
        weights = np.ones(len(y_true))
        if rule == "censor_ips":
            raw = 1.0 / np.maximum(propensity[reviewed], 1e-6)
            weights[reviewed] = np.minimum(raw, config.weight_clip * raw.mean())
        pool.add(X[keep], recorded[keep], weights[keep], y_true[keep])
        return

    if rule == "full_labels":
        pool.add(X, y_true, np.ones(len(y_true)), y_true)
        return

    if rule == "reviewed_only":
        pool.add(X[reviewed], y_true[reviewed], np.ones(len(reviewed)), y_true[reviewed])
        return

    if rule == "ips":
        # Only reviewed rows carry a trustworthy label, and each stands in for
        # 1/propensity transactions like it. Rows the policy was certain to
        # review get a weight of one and contribute no variance.
        weights = 1.0 / np.maximum(propensity[reviewed], 1e-6)
        ceiling = config.weight_clip * weights.mean()
        weights = np.minimum(weights, ceiling)
        pool.add(X[reviewed], y_true[reviewed], weights, y_true[reviewed])
        return

    if rule == "assume_negative":
        recorded = np.zeros(len(y_true), dtype=np.int8)
        recorded[reviewed] = y_true[reviewed]
        pool.add(X, recorded, np.ones(len(y_true)), y_true)
        return

    if rule == "soft_censor":
        recorded = np.zeros(len(y_true), dtype=np.int8)
        recorded[reviewed] = y_true[reviewed]
        # The floor keeps a confidently-scored fraud from vanishing entirely,
        # which would make the pool smaller in a way that depends on how wrong
        # the model already is.
        weights = np.maximum(1.0 - np.asarray(scores, dtype=float), config.soft_floor)
        weights[reviewed] = 1.0
        pool.add(X, recorded, weights, y_true)
        return

    if rule == "ips_augmented":
        recorded = np.zeros(len(y_true), dtype=np.int8)
        recorded[reviewed] = y_true[reviewed]
        weights = np.ones(len(y_true))
        raw = 1.0 / np.maximum(propensity[reviewed], 1e-6)
        weights[reviewed] = np.minimum(raw, config.weight_clip * raw.mean())
        pool.add(X, recorded, weights, y_true)
        return

    raise ValueError(f"unknown pool rule {rule!r}")


# --------------------------------------------------------------------------- #
# The loop itself.

# The detector is held fixed across every arm. This study is about what the
# model is taught, not about which model is best, and LightGBM is both the
# strongest baseline in our own comparison and cheap enough to refit once per
# round. Features go in unscaled because the trees are invariant to it.
DETECTOR_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
}


def balanced_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fold class balancing into the sample weights.

    Every arm gets the same treatment, so the comparison stays about the label
    bookkeeping. Doing it here rather than through ``class_weight`` also lets
    the inverse propensity weights compose with it instead of fighting it.
    """
    weights = np.asarray(w, dtype=float).copy()
    for label in (0, 1):
        mask = y == label
        count = int(mask.sum())
        if count:
            weights[mask] *= len(y) / (2.0 * count)
    return weights


def fit_detector(
    pool: LabelPool,
    seed: int,
    categorical: list[int] | None = None,
    overrides: dict | None = None,
):
    """Train the detector on the pool as the simulated bank knows it."""
    from lightgbm import LGBMClassifier

    X, y, w, _ = pool.arrays()
    params = {**DETECTOR_PARAMS, **(overrides or {})}
    model = LGBMClassifier(**params, random_state=seed)

    # A pool with one class in it cannot train a classifier. This happens when
    # a starved arm has gone several rounds without an analyst finding fraud,
    # and it is itself a symptom worth recording rather than an error.
    if len(np.unique(y)) < 2:
        return None

    model.fit(
        X, y,
        sample_weight=balanced_weights(y, w),
        categorical_feature=categorical or "auto",
    )
    return model


def score_round(model, X: np.ndarray) -> np.ndarray:
    """Fraud scores for a round, falling back to zeros when no model exists."""
    if model is None:
        return np.zeros(len(X), dtype=float)
    return model.predict_proba(X)[:, 1]


def run_simulation(
    X: np.ndarray,
    y: np.ndarray,
    round_index: np.ndarray,
    config: FeedbackConfig,
    *,
    warmup_rounds: int = 1,
    budget_fraction: float = 0.005,
    seed: int = 0,
    categorical: list[int] | None = None,
) -> list[RoundOutcome]:
    """Replay the stream one round at a time and record what each round cost.

    Rows must already be in chronological order and ``round_index`` says which
    retraining cycle each belongs to. The warm-up rounds are fully labelled for
    every arm: a bank switching on a new model has an audited pilot to start
    from, and giving every arm the same starting point means later differences
    come from the loop rather than from initialisation.
    """
    import warnings

    from sklearn.exceptions import UndefinedMetricWarning

    from src.common.metrics import evaluate

    rng = np.random.default_rng(seed)
    labels = np.unique(round_index)
    warmup, replay = labels[:warmup_rounds], labels[warmup_rounds:]

    pool = LabelPool()
    seen = np.isin(round_index, warmup)
    pool.add(X[seen], y[seen], np.ones(int(seen.sum())), y[seen])

    outcomes: list[RoundOutcome] = []
    for index, label in enumerate(replay):
        current = round_index == label
        X_round, y_round = X[current], y[current]
        budget = max(1, int(round(len(y_round) * budget_fraction)))

        model = fit_detector(pool, seed, categorical, config.detector)
        pool_size, pool_positives, pool_wrong = pool.stats()
        scores = score_round(model, X_round)

        reviewed, propensity = _select_reviews(scores, budget, config, rng)
        caught = int(y_round[reviewed].sum())

        # A round can contain no fraud at all, on ULB especially, and the
        # ranking metrics are then undefined rather than wrong. That is a
        # property of the stream, not a fault to be reported once per round.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            warnings.simplefilter("ignore", UserWarning)
            metrics = evaluate(y_round, scores, threshold=0.5, k=budget)
        present = int(y_round.sum())
        outcomes.append(
            RoundOutcome(
                round_index=index,
                n_transactions=len(y_round),
                n_frauds=present,
                n_reviewed=len(reviewed),
                frauds_caught=caught,
                auprc=metrics["auprc"],
                roc_auc=metrics["roc_auc"],
                recall_at_budget=caught / present if present else 0.0,
                precision_at_budget=caught / len(reviewed) if len(reviewed) else 0.0,
                pool_size=pool_size,
                pool_positives=pool_positives,
                pool_mislabelled=pool_wrong,
            )
        )

        update_pool(
            pool, X_round, y_round, reviewed, propensity, config,
            scores=scores, budget=budget,
        )

    return outcomes


def summarise(outcomes: list[RoundOutcome]) -> dict[str, float]:
    """Collapse a run to the numbers the paper reports.

    Frauds caught is summed rather than averaged because it is a count of real
    losses prevented over the whole simulated deployment. The ranking metrics
    are averaged over rounds, and reported again for the last round alone,
    since a loop that degrades slowly shows the damage at the end.
    """
    caught = sum(o.frauds_caught for o in outcomes)
    present = sum(o.n_frauds for o in outcomes)
    reviewed = sum(o.n_reviewed for o in outcomes)
    return {
        "frauds_caught": caught,
        "frauds_present": present,
        "reviews_spent": reviewed,
        # Reviews per fraud in the stream. This, rather than the review budget
        # as a share of transactions, is what governs how hard the loop bites,
        # and it is the axis the results are reported against.
        "capacity_ratio": reviewed / present if present else 0.0,
        "catch_rate": caught / present if present else 0.0,
        "review_precision": caught / reviewed if reviewed else 0.0,
        "auprc_mean": float(np.mean([o.auprc for o in outcomes])),
        "auprc_final": outcomes[-1].auprc,
        "auprc_first": outcomes[0].auprc,
        "auprc_drift": outcomes[-1].auprc - outcomes[0].auprc,
        "roc_auc_mean": float(np.mean([o.roc_auc for o in outcomes])),
        "recall_at_budget_mean": float(np.mean([o.recall_at_budget for o in outcomes])),
        "pool_mislabelled_final": outcomes[-1].pool_mislabelled,
        "pool_size_final": outcomes[-1].pool_size,
    }
