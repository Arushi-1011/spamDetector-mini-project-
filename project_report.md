# Spam Detector — Predictive Analytics Mini Project

## Objective
Build a supervised machine-learning system that classifies SMS text as ham (legitimate) or spam.

## Dataset
The project uses the UCI SMS Spam Collection. The original corpus contains 5,572 labelled records. After removing exact duplicate rows and invalid/empty records, 5,158 messages remain: 4,516 ham and 642 spam.

## Methodology
1. Load and clean the labelled SMS records.
2. Remove exact duplicate rows.
3. Encode ham as 0 and spam as 1.
4. Split into 80% training and 20% test data using stratification and random_state=42.
5. Convert text to TF-IDF features using unigrams and bigrams.
6. Train Logistic Regression with balanced class weights.
7. Evaluate on the held-out test set.
8. Save the complete TF-IDF + Logistic Regression pipeline for the Streamlit app.

## Results
See `metrics.json` for the exact run results. With the included fixed split, the model achieved approximately 97.97% accuracy, 90.84% spam precision, 92.97% spam recall, 91.89% spam F1-score, and 99.14% ROC-AUC. Confusion matrix format is [[TN, FP], [FN, TP]]: [[892, 12], [9, 119]].

These results are from one hold-out split and are not a guarantee of performance on future messages.

## Application
Run `streamlit run app.py` to launch the interface. Enter a message to see the predicted class and estimated spam probability.

## Limitations and ethics
The dataset is an older SMS corpus and may not represent current scam language, languages, or messaging platforms. A false negative may let a spam message pass; a false positive may incorrectly flag a legitimate message. The probability is an estimate, not a guarantee. Do not rely on the classifier alone for security decisions.
