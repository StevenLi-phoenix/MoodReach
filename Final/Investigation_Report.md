# Final Project Investigation Report

Prepared Nov 21, 2026 for team MoodReach (Section 2) ahead of the Dec 3 final presentation. Covers what the final must contain, why the dataset should change, which real datasets fit, and how far the methods should go given what the bootcamp taught.

## 1. What the final requires

Sources: Brightspace "Group Project" page, syllabus, announcements of Oct 29, Nov 2, Nov 19 and Nov 20.

- **When:** Thu Dec 3, 1:00–3:00 PM ET. We present to industry judges for **at most 5 minutes, including Q&A**. The order will be shared before the session.
- **Who:** Every member must attend, but not everyone has to present. Finishing the project is required for the badge.
- **Content:** "Present methodology, results and insights" (Week 9). The suggested timeline for our topic after the midterm is:
  - Week 5: initial **sentiment classification** model
  - Weeks 6–7: **hypothesis testing** on sentiment trends, engagement and platform differences
  - Week 8: final predictive model and interpretation
- **Judging (1–3 each):**
  - Project Design: is the project systematic and driven by proactive questions?
  - Addressing the Stated Problem: do the results answer the question?
  - Presentation & Demo: is it clear and concise?
- **Check-in form:** submitted Nov 21 with all four members and this repo.

## 2. Mid-program feedback and the dataset problem

At the Oct 29 midterm the judges asked what the end goal was, and they raised two problems with the data: it looks synthetic, and it is small.

The evidence supports them. The Kaggle set `kashishparmar02/social-media-sentiments-analysis-dataset` has about 732 rows (707 after cleaning). Its post times fall unnaturally within 06:00–24:00. One public GitHub project states the data was generated with ChatGPT, although the author never confirms this.

Any model trained on it will find almost no real signal, which hurts the "Addressing Stated Problem" score.

## 3. Real alternatives (text posts only)

We searched Kaggle and the Hugging Face Hub and checked each schema through the HF datasets-server. Video datasets are excluded.

| Dataset | Rows | Engagement | Sentiment / emotion | License | Fit |
|---|---|---|---|---|---|
| [paakhim10 tweets-and-engagement-metrics](https://huggingface.co/datasets/haoxianc/paakhim10_tweets-and-engagement-metrics) | 102,062 real tweets | Likes, RetweetCount, Reach, Klout | numeric Sentiment score (−6 … 7.3) | MIT | **Best drop-in**: same question, same feature types (Country ×151, Weekday, Hour, Gender, Lang, text) |
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | 58k Reddit comments | none | 27 emotions + neutral | Apache-2.0 | Train an emotion classifier and apply it to the tweets |
| [Twitter US Airline Sentiment](https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment) | 14,640 | retweet_count | 3 classes | CC BY-NC-SA 4.0 | Small fallback with real labels |
| [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) | 1.6M tweets | none | polarity | unverified | Polarity classifier only |
| [Arctic Shift Reddit](https://huggingface.co/datasets/Dk587/arctic) | very large | score, num_comments | none | Reddit ToS | Real engagement, but needs filtering and labeling |

Several HF hits are explicitly synthetic (for example `africa-synth-*`), or have only 2,000 rows of round numbers. We skipped them.

**Caveats for paakhim10:** the tweets lean toward tech and cloud topics (AWS, Azure), and the median Likes is 0. We should use log(RetweetCount + 1) or Reach as the engagement target and mention the topic skew as a limitation.

## 4. What the bootcamp taught (level calibration)

| Week | Topic | Techniques |
|---|---|---|
| 2 | NumPy, pandas | groupby/agg, merge, apply, pd.cut, missing values |
| 3 | EDA | hist, boxplot, corr, heatmap, log scale |
| 4 | SQL | joins, CTEs, window functions |
| 6 | ML foundations | train_test_split, LinearRegression, StandardScaler, confusion matrix, precision/recall, perceptron |
| 7 | Probability, Naive Bayes | scipy.stats, GaussianNB/BernoulliNB/MultinomialNB with CountVectorizer |
| 8 | Linear and logistic regression | gradient descent by hand, MAE/RMSE/R², LogisticRegression, polynomial overfitting |

- **In scope:** everything in the table, plus a baseline comparison.
- **Reasonable stretch:** TF-IDF, k-fold cross-validation, F1, Ridge/Lasso, and one t-test or chi-square test to back an EDA finding.
- **Beyond the course:** BERT/transformers, deep learning and XGBoost. These would be hard to explain in 5 minutes and would look out of step with the course.

## 5. Proposed plan

1. **Goal (one sentence):** predict how far a post reaches and identify what drives it (sentiment, time, place, author signals).
2. **Data:** switch to the paakhim10 tweets. Keep a short slide on why we left the synthetic set; this answers the judges' feedback directly.
3. **EDA:** rerun the key midterm views on real data (sentiment distribution, engagement by hour and weekday, country and language), and keep the best 3 findings.
4. **Sentiment classifier:** use CountVectorizer (optionally TF-IDF) with MultinomialNB, compared against LogisticRegression. Predict the sentiment class (negative/neutral/positive, binned from the score) from the text. Report accuracy, precision/recall and a confusion matrix.
5. **Hypothesis tests:**
   - Do positive and negative tweets get different retweets? Use a t-test or Mann–Whitney.
   - Is sentiment independent of weekday or country? Use chi-square.
6. **Engagement model:** use LinearRegression on log(retweets + 1), with features for sentiment, hour, weekday, Klout, IsReshare and country (top-k one-hot). Compare against a mean baseline and report MAE/RMSE/R². Optionally add LogisticRegression for "high reach" (top quartile) and show coefficients as the "drivers" chart.
7. **Bias–variance and limitations:** cover train vs test error, the topic skew, the time period (~2016), and why reach is noisy.
8. **Deliverables:**
   - `Final/final_analysis.ipynb`
   - figures in `Final/figures/`
   - 6–7 slides: problem → data → 3 insights → models → drivers → limits
   - an updated README

### Possible task split

| Person | Task |
|---|---|
| A | Data loading and cleaning, EDA rerun |
| B | Sentiment classifier |
| C | Hypothesis tests and engagement model |
| D | Slides and README; rehearse to 5 minutes |
