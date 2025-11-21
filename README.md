# DA5401_Final

# DA5401 Data Challenge: Metric Alignment via Ensemble Learning

*Student:* Venkata Sai Vishveswar  
*Roll No:* BE22B042  
*Course:* DA5401 (Data Science)

---

## 📌 Project Overview

Evaluating Generative AI is hard. While human evaluation is the gold standard, it is unscalable. This project aims to build an *Automated Metric Learning Model* that predicts a "Fitness Score" (0-10) representing how well an AI's response adheres to a specific evaluation metric (e.g., "Safety," "Fluency," "Code Quality").

*The Core Challenge:* The provided dataset was not a standard regression problem. It was heavily skewed, with *91% of samples scoring between 9 and 10*. The model needed to learn to detect "anomalies" (bad responses) rather than just predicting the mean.

---

## 🚀 Key Highlights

*   *Hybrid Feature Engineering:* Combined Deep Semantic Embeddings (MPNet) with Psycholinguistic Features (TextBlob) to capture both meaning and tone.
*   *Synthetic Data Augmentation:* Solved the class imbalance problem by generating "Synthetic Negatives" via Metric Mismatching.
*   *Stacked Generalization:* A robust 2-level ensemble (CatBoost + LightGBM + Random Forest) that outperforms individual models.
*   *In-Depth Bias Analysis:* Verified cross-lingual performance using custom visualization techniques.

---

## 📊 Exploratory Data Analysis (EDA)

My analysis revealed three critical insights that shaped the final model:

1.  *The "Perfect Score" Cliff:* A distribution analysis showed that scores $<8$ are statistical outliers. Standard MSE loss functions fail here.
2.  *Language Agnosticism:* Using a *Lollipop Bias Chart*, I determined that the LLM judge does not penalize non-English languages (Hindi, Tamil) significantly.
3.  *Tone Matters:* A scatter plot of *Subjectivity vs. Polarity* revealed that the judge prefers neutral, objective facts over opinionated responses. This motivated the inclusion of linguistic features.

---

## 🧪 Methodology

### 1. The Failed Experiment: Triplet Loss
I initially attempted a Deep Learning approach using *Siamese Networks and Triplet Loss* ($L = max(d(A,P) - d(A,N) + \alpha, 0)$).
*   *Outcome:* Failed.
*   *Reason:* "Vanishing Loss." Due to the scarcity of hard negatives, the model learned trivial distinctions too quickly and stopped learning meaningful semantic features.

### 2. The Pivot: Augmentation & Stacking
To overcome the data limitations, I adopted a feature-engineering heavy approach:

*   *Step A: Synthetic Negative Sampling:*
    I took high-quality responses (Score 10) and paired them with random, unrelated metrics. I assigned these pairs low scores (0-3). This taught the model that *Context is King*—a good response is only good if it answers the specific metric asked.

*   *Step B: Hybrid Features:*
    *   *Semantics:* paraphrase-multilingual-mpnet-base-v2 (State-of-the-art sentence embeddings).
    *   *Linguistics:* Extracted Sentiment Polarity, Subjectivity, and Lexical Diversity.

*   *Step C: The Stacked Ensemble:*
    I trained a meta-learner (Linear Regression) on top of three diverse base models:
    1.  *CatBoost:* Handles categorical metric names best.
    2.  *LightGBM:* Provides speed and gradient boosting power.
    3.  *Random Forest:* Crucial for handling the "outliers" (the low scores) which boosting methods sometimes miss.

---

## 🛠️ Installation & Usage

### Requirements
txt
pandas
numpy
scikit-learn
sentence-transformers
catboost
lightgbm
textblob
matplotlib
seaborn
langdetect


### Running the Notebook
1.  Ensure train_data.json, test_data.json, and metric_names.json are in the root directory.
2.  Run the notebook cells sequentially.
    *   *Note:* The generate_hybrid_features function uses MPNet and may take a few minutes on CPU.
    *   *Note:* The Stacking Regressor uses 5-Fold CV, so training logs will repeat 5 times.

---

## 📈 Results

The Stacked Ensemble demonstrated superior stability compared to individual models.

| Model | Approach | Key Strength |
| :--- | :--- | :--- |
| *Triplet Network* | Deep Metric Learning | Theoretically sound, failed in practice due to data sparsity. |
| *Single LightGBM* | Gradient Boosting | Fast, but struggled with the "perfect score" bias. |
| *Stacked Ensemble* | *Final Solution* | *Best Performance.* RF captured outliers, CatBoost captured semantics. |

---

## 📜 Conclusion

This project demonstrates that in real-world Data Science, *Data-Centric AI* (augmenting data, engineering features) often beats *Model-Centric AI* (complex architectures). By synthesizing negative examples and explicitly modeling linguistic tone, we achieved a robust evaluation metric despite extreme class imbalance.
