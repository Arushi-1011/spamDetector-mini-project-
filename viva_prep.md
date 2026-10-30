# Viva Preparation — Spam Detector

**1. What is the objective?**  
To classify SMS messages into ham (normal) and spam (unwanted) using supervised machine learning.

**2. Which dataset is used?**  
The UCI SMS Spam Collection, containing 5,572 labelled records before duplicate removal.

**3. What is ham?**  
A legitimate, non-spam message.

**4. Why is preprocessing needed?**  
To clean labels and text, remove empty records and exact duplicate rows, and prepare consistent input for the model.

**5. What is TF-IDF?**  
Term Frequency–Inverse Document Frequency converts text into numeric features. It gives importance to terms that are frequent in a message but less common across the corpus.

**6. Why use unigrams and bigrams?**  
Unigrams capture individual words; bigrams capture two-word phrases such as “free entry” or “call now”.

**7. Why Logistic Regression?**  
It is a fast and effective baseline for binary text classification and provides class probabilities.

**8. Why use a Pipeline?**  
It keeps vectorization and classification together, ensuring the app applies the same preprocessing used during training.

**9. Why stratified train-test split?**  
Spam is the minority class. Stratification keeps similar ham/spam proportions in train and test sets.

**10. Explain precision and recall.**  
Spam precision measures how many messages predicted as spam are actually spam. Spam recall measures how many actual spam messages the model detects.

**11. What is F1-score?**  
The harmonic mean of precision and recall.

**12. What is a confusion matrix?**  
For this binary setup, [[TN, FP], [FN, TP]]: true ham predicted ham, ham wrongly predicted spam, spam missed as ham, and spam correctly detected.

**13. Why can accuracy be misleading?**  
Because most messages are ham. A model could achieve high accuracy while missing many spam messages, so precision, recall and F1 are also important.

**14. What does spam probability mean?**  
It is the model's estimated probability-like score for the spam class, not a guarantee that a message is malicious.

**15. What are the limitations?**  
The dataset is older and relatively small; new scam patterns, languages and messaging styles may reduce performance. False positives and false negatives are possible.
