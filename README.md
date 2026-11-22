# AI-Based Predictive Maintenance and Fault Diagnosis of Machines

This project applies machine learning to predict machine failure (failure / no failure) from sensor data, and compares three algorithms on two datasets.

## Datasets
- **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository, S. Matzka, 2020): 10,000 synthetic records, about 3.4% failures (highly imbalanced). Features: air temperature, process temperature, rotational speed, torque, tool wear, product type.
- **Machine Downtime Dataset**: about 2,500 records of machining sensor readings, almost evenly balanced. Source: [add platform / link here].

## Methodology
- **Machine Downtime:** target encoded as binary (1 = failure), non-sensor columns dropped, invalid non-positive sensor readings treated as missing, data sorted by date and split chronologically 80/20, missing values filled with training-set medians.
- **AI4I 2020:** identifier and failure-mode columns (TWF, HDF, PWF, OSF, RNF) removed to avoid target leakage, `Type` one-hot encoded, stratified 80/20 split, class-weight balancing (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost).
- **Models:** Logistic Regression (with StandardScaler), Random Forest, XGBoost.
- **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices, ROC curves, feature importance.

## Results

**Machine Downtime**

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.886 | 0.881 | 0.885 | 0.883 | 0.951 |
| Random Forest | 0.994 | 0.992 | 0.996 | 0.994 | 1.000 |
| XGBoost | 0.992 | 0.984 | 1.000 | 0.992 | 1.000 |

**AI4I 2020**

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.824 | 0.142 | 0.824 | 0.242 | 0.907 |
| Random Forest | 0.982 | 0.919 | 0.500 | 0.648 | 0.960 |
| XGBoost | 0.964 | 0.483 | 0.853 | 0.617 | 0.971 |

On AI4I, accuracy is misleading because always predicting "no failure" already scores about 96.6%, so Recall, F1-score and ROC-AUC matter more.

## Repository structure
```
.
├── notebook.ipynb        # main Jupyter/Colab notebook (rename to match yours)
├── data/                 # datasets (or links, see above)
├── images/               # saved graphs (.png)
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
```
Open the notebook and run all cells in order.

## Limitations and future work
- Offline analysis on historical data, not connected to live sensors.
- AI4I is synthetic; Machine Downtime classes are almost perfectly separable, so scores may not carry over to noisier real data.
- Only binary failure prediction; identifying the failure type (fault diagnosis) is future work.
- Default 0.5 threshold; hyperparameter tuning and cross-validation are future work.

## References
- S. Matzka, *AI4I 2020 Predictive Maintenance Dataset*, UCI Machine Learning Repository, 2020.
- L. Breiman, "Random Forests," *Machine Learning*, 2001.
- T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016.
- F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, 2011.
