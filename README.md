# Hiwet Breast Cancer AI — SVM Classification

An educational machine-learning project that classifies observations from the **Breast Cancer Wisconsin (Diagnostic)** dataset as malignant or benign using a Support Vector Machine (SVM).

## Project features

- Reproducible train/test split with stratification
- StandardScaler + RBF SVM
- 5-fold GridSearchCV using ROC-AUC
- Final held-out test evaluation
- Accuracy, precision, recall, F1 and ROC-AUC
- Confusion matrix and ROC curve
- Saved model, scaler and complete pipeline
- Streamlit dashboard for manual input
- CSV upload with strict column/value validation
- Benign/malignant demonstration examples
- Clear educational/medical disclaimer

## Model result

The current reproducible training run uses:

- Dataset: 569 rows, 30 features
- Test split: 20%
- Random state: 42
- Best SVM: `C=10`, `gamma=0.01`, `kernel=rbf`
- 5-fold CV ROC-AUC: approximately `0.9954`
- Held-out test ROC-AUC: approximately `0.9977`
- Held-out test accuracy: approximately `98.25%`

See `metrics.json` for the complete recorded results.

## Files

```text
Hiwet_Breast_Cancer/
├── app.py
├── train_model.py
├── breast_cancer_svm.ipynb
├── model.pkl
├── scaler.pkl
├── pipeline.pkl
├── metrics.json
├── sample_benign_features.csv
├── sample_malignant_features.csv
├── sample_template.csv
├── requirements.txt
└── README.md
```

## Run the Streamlit application

From this folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## Retrain the model

The dataset is built into scikit-learn, so no external dataset download is required.

```bash
python train_model.py
```

This regenerates `model.pkl`, `scaler.pkl`, `pipeline.pkl`, and `metrics.json`.

## Notebook

Open `breast_cancer_svm.ipynb` to see the complete workflow:

1. Load and inspect the dataset
2. Check missing/duplicate values and class distribution
3. Split into training and testing sets
4. Build a preprocessing + SVM pipeline
5. Tune hyperparameters with 5-fold cross-validation
6. Evaluate once on the untouched test set
7. Plot confusion matrix and ROC curve
8. Save reproducible artifacts

Using a pipeline is important because scaling happens inside each cross-validation fold, preventing preprocessing leakage during tuning.

## CSV format

The uploaded CSV must contain the exact 30 feature names used by scikit-learn's Breast Cancer Wisconsin Diagnostic dataset. The app checks:

- required columns
- unexpected columns
- empty files
- numeric values
- finite values

If the CSV has multiple rows, the app uses the first row for this demonstration.

## Important limitation

This is an **educational machine-learning demonstration**, not a medical diagnostic system. Model-estimated probabilities are not clinical certainty. Real diagnosis and treatment decisions require qualified healthcare professionals and appropriate clinical testing.
