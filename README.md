# Spam Detector — Predictive Analytics Mini Project

A working SMS spam classifier built with the **UCI SMS Spam Collection** dataset, TF-IDF text features, and Logistic Regression. It includes an executed Jupyter notebook, a reusable training script, a saved model, and a Streamlit interface.

## Dataset
- Source: UCI Machine Learning Repository, SMS Spam Collection (Dataset 228).
- Included file: `spam.csv` (5,158 unique messages after removing exact duplicate rows; 4,516 ham and 642 spam).
- Labels: `ham` = legitimate message; `spam` = unwanted message.
- Original corpus file: `SMSSpamCollection`.

## Run locally
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open the local URL printed by Streamlit (normally http://localhost:8501).
## Live Demo 
https://smspamdetector.streamlit.app

## Notebook
Open `spam_classifier.ipynb` in Jupyter Notebook or VS Code and run the cells from top to bottom. It covers dataset inspection, class distribution, preprocessing, TF-IDF, Logistic Regression, evaluation, and example predictions.

## Model workflow
1. Load and clean the labelled SMS messages.
2. Convert ham/spam labels to 0/1.
3. Split data into stratified 80% training and 20% testing subsets.
4. Convert text to numerical TF-IDF unigram/bigram features.
5. Train Logistic Regression with balanced class weights.
6. Evaluate on the held-out test set.
7. Save the complete preprocessing + classifier pipeline in `spam_classifier_model.joblib`.

## Important evaluation note
The metrics in `metrics.json` are from a single stratified hold-out split with `random_state=42`. Results depend on the split and configuration. Do not alter the test labels or tune repeatedly against the test set just to force a target accuracy. Report the actual results produced by the training script.

## Files
- `spam_classifier.ipynb` — notebook with explanations and visualizations
- `app.py` — Streamlit user interface
- `train_model.py` — reproducible model training and evaluation
- `spam_classifier_model.joblib` — fitted TF-IDF + Logistic Regression pipeline
- `spam.csv` — cleaned dataset
- `metrics.json` — measured test-set metrics
- `project_report.md` — concise project report
- `viva_prep.md` — viva questions and answers
- `SMSSpamCollection` — original UCI text corpus
- `readme` — original dataset notes
