# Temporal Robustness of Machine Learning-Based Flight Delay Prediction

### A Case Study at Istanbul Airport (IST)

**Master's Thesis — MSc Big Data Analytics & Management**  
Bahçeşehir University · Istanbul, June 2026

**Author:** Sheraz Lyna Touhouche  
**Supervisor:** Assoc. Prof. Tevfik Aytekin

---

## Quick Access

- [Read the Master's Thesis](report/LYNA_TOUHOUCHE_THESIS2026.pdf)
- [View the Defense Presentation](presentation/thesis_defense_presentation.pdf)
- [Explore the Source Code](scripts/)
- [Explore Experimental Results](results/plots/)
- [View Additional Visualizations](visu/)

---

## 1. Research Overview

Flight delay prediction is a challenging problem in aviation, where operational conditions, weather patterns, airline schedules, and interconnected flight rotations influence departure performance.

This research investigates the **temporal robustness of machine learning models for flight delay prediction**, using Istanbul Airport (IST/LTFM) as a case study.

Rather than focusing exclusively on predictive accuracy, the study examines whether models trained on historical operational data can generalize to future periods under changing seasonal conditions.

The central research concern is that strong performance under conventional random train-test splits may not translate into reliable real-world forecasting when the underlying feature–target relationships evolve over time.

The research combines aviation, meteorological, and surface traffic information to investigate predictive performance, seasonal distribution shifts, delay propagation, and model robustness.

## 2. Research Questions

The investigation is structured around four research questions:

- **RQ1 — Predictive Performance:** How accurately can machine learning models predict flight delays at Istanbul Airport, and what is the relative influence of different feature categories?
- **RQ2 — Temporal Generalization:** Do predictive relationships learned during autumn remain valid when models are evaluated on subsequent winter operations?
- **RQ3 — Performance Degradation:** What mechanisms drive temporal performance degradation, and which modelling strategies can mitigate it?
- **RQ4 — Surface Traffic:** Does the inclusion of road traffic variables provide meaningful predictive signal for departure delays?

## 3. Dataset and Study Scope

| Component | Description |
|---|---|
| Airport | Istanbul Airport (IST / LTFM) |
| Study period | October 2025 – February 2026 |
| Raw API records | 42,623 |
| Final analytical dataset | 9,527 flights |
| Prediction target | Departure delay ≥ 15 minutes |
| Overall delay rate | 79.4% |
| Training period | October–November 2025 |
| Intermediate period | December 2025 |
| Temporal test period | January–February 2026 |

### Data Sources

The study integrates three categories of information:

- **Aviation data:** Flight records and operational information.
- **Meteorological data:** Hourly weather observations, including temperature, humidity, pressure, and wind speed.
- **Surface traffic data:** Road speed, vehicle count, and congestion indicators for airport access routes.

Raw datasets and credentials are not included in this repository.

See [`data/README.md`](data/README.md) for information about the data sources and data organization.

## 4. Methodology

The research follows a structured machine learning pipeline:

1. Data collection and integration across heterogeneous sources.
2. Data cleaning, preprocessing, and missing-value treatment.
3. Chronological feature engineering and temporal leakage prevention.
4. Model development and hyperparameter optimization.
5. Random-split and chronological evaluation.
6. Model explainability and feature importance analysis.
7. Distribution shift and temporal robustness investigation.
8. Robustness experiments and exploratory cost-sensitive analysis.

### Feature Engineering

Five feature categories were investigated:

- **Historical delays:** Previous-hour delay rate, rolling 24-hour delay rate, and airline-specific rolling delay rates.
- **Weather:** Temperature, humidity, pressure, wind speed, and weather categories.
- **Airline identity:** Carrier-level categorical information.
- **Surface traffic:** Average road speed, vehicle count, and congestion index.
- **Temporal features:** Hour of day and day of week.

Strict chronological feature construction was implemented to prevent information from future flights or the target flight from leaking into historical predictors.

## 5. Machine Learning Models

Three algorithms were evaluated:

| Model | Role |
|---|---|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Ensemble learning through bagging |
| XGBoost | Gradient-boosted decision trees |

Hyperparameter optimization was performed for Random Forest and XGBoost using randomized search and cross-validation within the training period.

Model evaluation primarily relies on ROC-AUC, alongside additional classification metrics and model interpretation techniques.

## 6. Temporal Validation: The Central Finding

A major contribution of this research is the comparison between conventional random-split evaluation and chronological testing on future operational periods.

### Random-Split Performance

Under conventional random-split evaluation, the tuned XGBoost model achieved:

- **ROC-AUC: 0.903**
- Recall: 0.941
- Precision: 0.865

These results demonstrate strong predictive discrimination when training and testing observations are drawn from a mixed distribution.

### Temporal Performance

When evaluated chronologically, performance deteriorated substantially:

| Evaluation period | ROC-AUC |
|---|---:|
| Random split | 0.903 |
| December 2025 | 0.514 |
| January–February 2026 | 0.474 |

The results reveal a significant gap between conventional offline evaluation and forecasting performance on subsequent seasonal data.

The degradation was observed across all three model architectures, suggesting that the problem is not limited to a particular algorithm.

## 7. Concept Drift and Delay Propagation

The study investigates changes in the relationships between historical delay features and subsequent flight delays.

One of the central findings concerns the seasonal reversal in delay propagation:

- **Autumn (October–November):** +7.07 percentage points.
- **December:** +17.10 percentage points.
- **Winter (January–February):** −7.78 percentage points.

This represents a substantial change in the observed propagation relationship across the study periods.

The analysis suggests that historical delay features, which were highly informative during training, may become unreliable or even misleading under different seasonal operating conditions.

Feature-level distribution analysis and SHAP explainability provide additional evidence consistent with temporal concept drift.

## 8. Explainability and Feature Importance

SHAP (SHapley Additive exPlanations) was used to investigate the contribution of individual features and feature categories to model predictions.

The reported feature-category contributions were:

| Feature category | SHAP importance |
|---|---:|
| Historical delays | 40.6% |
| Weather | 33.9% |
| Airline identity | 10.2% |
| Surface traffic | 8.9% |
| Temporal features | 6.4% |

The results highlight the importance of historical delay propagation and meteorological variables in the model's predictive behaviour.

They also reveal a potential vulnerability: the model relies heavily on historical delay features that undergo substantial distribution shifts between training and test periods.

## 9. Robustness Experiments

Six modelling strategies were investigated to explore possible ways of mitigating temporal degradation.

| Strategy | January–February AUC |
|---|---:|
| Global XGBoost baseline | 0.474 |
| Expanding window | 0.566 |
| Sliding window (2 months) | 0.573 |
| Sliding window (1 month) | 0.569 |
| Season-aware Random Forest | 0.599 |
| Propagation removal (season-aware) | 0.624 |

The best result was obtained by removing historical delay propagation features within the season-aware framework.

Although this improved performance compared with the global baseline, temporal robustness remained limited. The results therefore suggest a promising direction rather than a deployment-ready solution.

## 10. Surface Traffic Analysis

The contribution of surface road traffic variables was assessed through correlation analysis, feature ablation, and SHAP importance.

The results indicated limited predictive value under the available data conditions:

- Mean absolute Pearson correlation: **0.03**
- AUC change after removing traffic features: **−0.002**
- Combined SHAP importance: **8.9%**

An important limitation is that traffic variables were constructed from 24-hour averaged profiles rather than actual day-specific traffic observations.

Consequently, the findings do not establish that real-time traffic information is uninformative. Further research using daily or real-time traffic feeds remains necessary.

## 11. Tools and Technologies

- **Programming:** Python
- **Data processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainable AI:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Data sources:** Aviation, meteorological, and urban traffic APIs

## 12. Repository Structure

```text
thesis/
├── data/                  # Raw datasets (excluded from Git)
├── scripts/
│   ├── data_collection/   # Data acquisition
│   ├── features/          # Feature engineering
│   └── models/            # Training and evaluation
├── results/
│   └── plots/             # Selected experimental results
├── report/
│   ├── LYNA_TOUHOUCHE_THESIS2026.pdf
│   └── main.tex
├── presentation/
│   └── thesis_defense_presentation.pdf
├── visu/                  # Additional visualizations
├── VISUALIZATION_GUIDE.md
└── README.md
```

## 13. Limitations and Future Research

The findings should be interpreted within the boundaries of the study:

- The research focuses on a single airport and one autumn-to-winter transition.
- The temporal test set contains 433 flights.
- Approximately 77% of raw API records were excluded, primarily because actual departure timestamps were unavailable.
- The Missing Completely At Random assumption was not formally tested.
- Surface traffic data relies on averaged profiles rather than real-time observations.
- Delay-cause codes were unavailable, limiting causal interpretation of propagation mechanisms.
- The seasonal modelling strategies require further validation across additional years and airports.

Future research could investigate full annual operational cycles, multi-airport datasets, real-time traffic integration, online learning, continuous retraining, and more adaptive approaches to concept drift.

## 14. Main Research Contributions

This thesis makes several empirical and methodological contributions:

1. Demonstrates the importance of chronological evaluation when assessing flight delay prediction models.
2. Quantifies substantial temporal performance degradation between random-split and future-period evaluation.
3. Documents seasonal changes in flight delay propagation patterns.
4. Investigates airline-specific heterogeneity in delay contagion.
5. Empirically evaluates the predictive contribution of surface road traffic data.
6. Implements a temporal validation framework with strict leakage prevention.
7. Explores season-aware modelling and feature removal as potential robustness strategies.

## 15. Conclusion

The main conclusion is that **strong offline predictive performance is necessary but not sufficient for reliable real-world deployment**.

The observed decline from a random-split ROC-AUC of 0.903 to 0.474 under chronological testing highlights the importance of temporal robustness in aviation machine learning.

By investigating seasonal distribution shifts, delay propagation, model explainability, and alternative modelling strategies, this research contributes to a more critical understanding of how predictive systems behave under changing operational conditions.

---

## Author

**Sheraz Lyna Touhouche**  
MSc Big Data Analytics & Management  
Bahçeşehir University

Research interests: Machine Learning · Data Science · Artificial Intelligence · Predictive Modelling · Explainable AI · Temporal Data Analysis

**Supervisor:** Assoc. Prof. Tevfik Aytekin  
**Location:** Istanbul, Türkiye  
**Year:** 2026