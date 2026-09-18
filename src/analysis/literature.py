"""Published results used for the "comparison with prior work" tables.

Every entry was read from the paper or repository named in ``source`` and is
stored with the evaluation protocol that produced it, because on these datasets
the protocol explains most of the spread. Numbers obtained by resampling before
the train/test split, or by any other route that lets test information reach the
model, are marked ``leaky`` - they are reported for context and are not treated
as targets to beat.

Comparisons like these are indicative, not exact: splits, seeds, thresholds and
feature handling differ between studies. That is precisely why every baseline in
both papers is also reimplemented and rerun here under one protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LiteratureResult:
    study: str
    year: int
    dataset: str
    model: str
    protocol: str
    metrics: dict[str, float]
    source: str
    leaky: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- #
# Paper 1: ULB / Worldline credit card fraud
# --------------------------------------------------------------------------- #
PAPER1_LITERATURE: list[LiteratureResult] = [
    LiteratureResult(
        study="Comparative Analysis of ML Models using SMOTE (IJSSE)",
        year=2025,
        dataset="ulb_creditcard",
        model="Random Forest + SMOTE",
        protocol="random stratified split, SMOTE applied to the dataset",
        metrics={"f1": 0.872, "roc_auc": 0.978, "auprc": 0.871},
        source="https://doi.org/10.18280/ijsse.150504",
        leaky=True,
        notes="Resampling is not described as being confined to the training fold.",
    ),
    LiteratureResult(
        study="Comparative Analysis of ML Models using SMOTE (IJSSE)",
        year=2025,
        dataset="ulb_creditcard",
        model="XGBoost + SMOTE",
        protocol="random stratified split, SMOTE applied to the dataset",
        metrics={"f1": 0.837, "roc_auc": 0.983, "auprc": 0.867},
        source="https://doi.org/10.18280/ijsse.150504",
        leaky=True,
    ),
    LiteratureResult(
        study="Comparative Analysis of ML Models using SMOTE (IJSSE)",
        year=2025,
        dataset="ulb_creditcard",
        model="Logistic Regression",
        protocol="random stratified split",
        metrics={"auprc": 0.724},
        source="https://doi.org/10.18280/ijsse.150504",
    ),
    LiteratureResult(
        study="Random Forest and XGBoost with and without SMOTE (IJAC)",
        year=2025,
        dataset="ulb_creditcard",
        model="Random Forest (no SMOTE)",
        protocol="random stratified split",
        metrics={"precision": 0.94, "recall": 0.82, "f1": 0.87, "mcc": 0.8763},
        source="https://revistas.uepg.br/index.php/ijac/article/view/25781",
    ),
    LiteratureResult(
        study="Random Forest and XGBoost with and without SMOTE (IJAC)",
        year=2025,
        dataset="ulb_creditcard",
        model="XGBoost + SMOTE",
        protocol="random stratified split",
        metrics={"precision": 0.79, "recall": 0.85, "f1": 0.82, "mcc": 0.8179, "roc_auc": 0.98},
        source="https://revistas.uepg.br/index.php/ijac/article/view/25781",
        leaky=True,
    ),
    LiteratureResult(
        study="Imbalance-Aware Evaluation and HPO (IJLTEMAS)",
        year=2026,
        dataset="ulb_creditcard",
        model="XGBoost + SMOTE (training fold only)",
        protocol="random stratified 80/20, SMOTE after the split, threshold tuned to 0.70",
        metrics={"auprc": 0.817, "roc_auc": 0.970, "precision": 0.81, "recall": 0.81, "f1": 0.81},
        source="https://doi.org/10.51583/ijltemas.2026.150400092",
        notes="Leakage-free resampling, but the split is random rather than chronological.",
    ),
    LiteratureResult(
        study="Imbalance-Aware Evaluation and HPO (IJLTEMAS)",
        year=2026,
        dataset="ulb_creditcard",
        model="Random Forest",
        protocol="random stratified 80/20, SMOTE after the split",
        metrics={"auprc": 0.805, "precision": 0.93},
        source="https://doi.org/10.51583/ijltemas.2026.150400092",
    ),
    LiteratureResult(
        study="Supervised and Ensemble Methods (ICCIAA)",
        year=2026,
        dataset="ulb_creditcard",
        model="Tree ensembles",
        protocol="stratified cross-validation, leak-free preprocessing pipeline",
        metrics={"mcc": 0.90, "auprc": 0.94},
        source="https://doi.org/10.1109/icciaa68481.2026.11544046",
        notes="Strongest published figures on this dataset; the split is still random.",
    ),
]


# --------------------------------------------------------------------------- #
# Paper 2: credit scoring
# --------------------------------------------------------------------------- #
PAPER2_LITERATURE: list[LiteratureResult] = [
    LiteratureResult(
        study="Yeh & Lien",
        year=2009,
        dataset="uci_dccc",
        model="Artificial neural network",
        protocol="original benchmark study",
        metrics={"roc_auc": 0.78},
        source="https://doi.org/10.1016/j.eswa.2007.12.020",
        notes="The dataset's originating paper; the reference point every later study cites.",
    ),
    LiteratureResult(
        study="Model-Risk-Friendly PD Workflow (JACS)",
        year=2024,
        dataset="uci_dccc",
        model="XGBoost",
        protocol="60/10/10/20 split with separate calibration partitions",
        metrics={"roc_auc": 0.7796, "auprc": 0.5526, "brier": 0.1351},
        source="https://doi.org/10.69987/jacs.2024.40606",
    ),
    LiteratureResult(
        study="LLM-Explanation-Enhanced Gradient Boosting (JACS)",
        year=2024,
        dataset="uci_dccc",
        model="XGBoost with engineered features",
        protocol="five repeated stratified splits",
        metrics={"roc_auc": 0.7943, "auprc": 0.5706},
        source="https://doi.org/10.69987/jacs.2024.40508",
        notes="Best reported mean ROC-AUC we found on this dataset under an honest protocol.",
    ),
    LiteratureResult(
        study="LLM-Explanation-Enhanced Gradient Boosting (JACS)",
        year=2024,
        dataset="uci_dccc",
        model="LightGBM with engineered features",
        protocol="five repeated stratified splits",
        metrics={"roc_auc": 0.7911, "auprc": 0.5686},
        source="https://doi.org/10.69987/jacs.2024.40508",
    ),
    LiteratureResult(
        study="Lessmann et al., benchmark study",
        year=2015,
        dataset="give_me_some_credit",
        model="Heterogeneous ensemble",
        protocol="large-scale benchmark across 41 classifiers",
        metrics={"roc_auc": 0.865},
        source="https://doi.org/10.1016/j.ejor.2015.05.030",
    ),
    LiteratureResult(
        study="Gunnarsson et al.",
        year=2021,
        dataset="give_me_some_credit",
        model="XGBoost",
        protocol="benchmark comparison of deep learning against gradient boosting",
        metrics={"roc_auc": 0.848},
        source="https://doi.org/10.1016/j.ejor.2021.03.006",
        notes="Concluded that gradient boosting still beats deep learning on credit data.",
    ),
    LiteratureResult(
        study="Improving ML Performance by Data Pre-Processing (SCITEPRESS)",
        year=2025,
        dataset="give_me_some_credit",
        model="Balanced Random Forest",
        protocol="grid search over resampling strategies",
        metrics={"roc_auc": 0.875},
        source="https://www.scitepress.org/Papers/2025/131185/",
        notes="Best figure we found on this dataset under an honest protocol.",
    ),
    LiteratureResult(
        study="NOTE: non-parametric oversampling (Scientific Reports)",
        year=2024,
        dataset="give_me_some_credit",
        model="Random Forest + NOTE oversampling",
        protocol="oversampling applied before evaluation",
        metrics={"roc_auc": 0.9837},
        source="https://doi.org/10.1038/s41598-024-78055-5",
        leaky=True,
        notes="Far above the honest ceiling of roughly 0.87; typical of resampling before the split.",
    ),
    LiteratureResult(
        study="Graph-Based Inductive Learning (Computational Economics)",
        year=2025,
        dataset="give_me_some_credit",
        model="GraphSAGE + CTGAN augmentation",
        protocol="synthetic augmentation before evaluation",
        metrics={"roc_auc": 0.9911, "f1": 0.9698},
        source="https://doi.org/10.1007/s10614-025-11114-9",
        leaky=True,
    ),
    LiteratureResult(
        study="Graph-Based Inductive Learning (Computational Economics)",
        year=2025,
        dataset="german_credit",
        model="GraphSAGE + CTGAN augmentation",
        protocol="synthetic augmentation before evaluation",
        metrics={"roc_auc": 0.8796, "f1": 0.8346},
        source="https://doi.org/10.1007/s10614-025-11114-9",
        leaky=True,
    ),
    LiteratureResult(
        study="Systematic Review of Credit Scoring Models (IJCRT)",
        year=2026,
        dataset="german_credit",
        model="Gradient boosting, pooled across studies",
        protocol="review of 28 studies, mean and spread",
        metrics={"roc_auc": 0.87},
        source="https://ijcrt.org/papers/IJCRT2603129.pdf",
        notes="Pooled mean 0.87 with standard deviation 0.06; individual studies span 0.78-0.96.",
    ),
]


# Honest performance ceilings, used to sanity-check our own runs and to argue in
# the text that figures well above them signal a protocol problem.
HONEST_CEILINGS = {
    "uci_dccc": {"roc_auc": (0.78, 0.82), "ks": (0.40, 0.45)},
    "give_me_some_credit": {"roc_auc": (0.85, 0.88)},
    "german_credit": {"roc_auc": (0.78, 0.82)},
    "ulb_creditcard": {"auprc": (0.80, 0.88)},
}


def for_dataset(dataset: str, paper: int) -> list[LiteratureResult]:
    table = PAPER1_LITERATURE if paper == 1 else PAPER2_LITERATURE
    return [entry for entry in table if entry.dataset == dataset]
