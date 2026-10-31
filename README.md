# Universal Plus Banking Prediction

*Predicting which customers will apply for business credit, following CRISP-DM*

A CRISP-DM-structured machine learning pipeline predicting which anonymised bank customers are likely to apply for business credit, tuned and validated across a 100,000-row dataset with a ~9:1 class imbalance — built for the *Analytics in Practice* module of Warwick Business School's MSc Business Analytics.

## Business problem

Universal Plus, a bank, wants to identify customers who are likely to apply for business credit in the near future, so it can target communications, guidance and product offers. Missing a likely applicant (a false negative) is the costlier mistake, so the model is explicitly optimised for **recall** (catching as many true applicants as possible) rather than raw accuracy, which is a poor guide here given how imbalanced the outcome is.

## Data

100,000 anonymised customers with 200 anonymised numeric features (`var_0`–`var_199`) plus a binary `target` (applied / did not apply). The target is heavily imbalanced: only 10.0% of customers applied (an 8.96:1 majority/minority ratio). **The dataset is not included in this repository** — it's a large (100k-row), non-redistributable coursework dataset. The notebook expects it at `datafile_full.csv` in the same directory; see [Running it](#running-it).

## Methodology (CRISP-DM)

- **Data understanding & quality checks** — duplicate rows/IDs, zero-variance features, missingness, and outlier presence (IQR-based) across all 200 anonymised features.
- **Data preparation** — dropped one high-missing column (`var_45`), median-imputed two others (`var_76`, `var_96`), dropped remaining incomplete rows (100,000 → 99,903 rows), then a stratified 80/20 train-test split (79,922 / 19,981 rows) to preserve the ~10% positive rate in both sets.
- **Modelling** — logistic regression with `class_weight="balanced"` (L2 penalty, `C=0.001`, liblinear solver), with hyperparameters originally tuned via grid/randomised search on a smaller representative sample for computational efficiency, then re-fit on the full, scaled (StandardScaler) dataset. Several other model families (decision tree, random forest, XGBoost, SVM) were evaluated during model selection; logistic regression was retained for the final pipeline for its balance of recall and interpretability.
- **Evaluation** — accuracy, precision, recall, F1, and ROC-AUC on held-out test data, plus a decision-threshold sweep (0.3 / 0.4 / 0.5) to show the precision-recall trade-off available to the business.
- **Explainability** — permutation feature importance (PFI), measured as the drop in **recall and AUC** when each feature is randomly shuffled — chosen deliberately over accuracy-based importance, since accuracy is a poor signal under this level of class imbalance.

## Key results

| Metric | Train | Test |
|---|---|---|
| Accuracy | 77.20% | 77.24% |
| Precision | 27.53% | 27.37% |
| Recall | 78.96% | **77.69%** |
| F1 | 0.408 | 0.405 |
| ROC-AUC | 0.862 | 0.853 |

Train/test metrics track closely, indicating the model generalises well rather than overfitting. At the default 0.5 threshold, the model recovers nearly 78% of true applicants. The threshold sweep shows the trade-off is tunable: lowering the threshold to 0.3 pushes recall to 91.2% (at the cost of precision dropping to 18.0%) — giving the business an explicit lever to trade contact volume against missed-applicant risk, depending on the relative cost of each.

Permutation feature importance identified `var_76`, `var_44`, and `var_53` as the features whose removal caused the largest drop in recall — the strongest signals for identifying likely applicants among the 200 anonymised variables.

## Repository structure

```
notebooks/
  credit_propensity_model.ipynb   # Full CRISP-DM pipeline: EDA, cleaning, modelling, evaluation, PFI
```

## Running it

The dataset isn't included (see [Data](#data) above) — place your own `datafile_full.csv` (same schema: `ID_code`, `target`, `var_0`...`var_199`) in the `notebooks/` directory, then:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/credit_propensity_model.ipynb
```

## Tech stack

Python · scikit-learn (Logistic Regression, Decision Tree, Random Forest, SVM) · XGBoost · pandas · NumPy · matplotlib / seaborn

## Attribution

**Team project.** Completed as group coursework (Group 8, Analytics in Practice, MSc Business Analytics, Warwick Business School). The code and analysis in this repository are the team's shared work, published with full credit to the group to show the project; this is not a claim of sole authorship. The dataset is anonymised, coursework-provided, and not redistributed here.

## License

MIT — see [LICENSE](LICENSE).
