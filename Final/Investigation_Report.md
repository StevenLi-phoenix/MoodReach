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

## 6. Results and deliverables (first full run)

The plan above is implemented in `final_analysis.ipynb`. Three decisions changed it after checking the data:

- **Original tweets only.** A reshare's `RetweetCount` is the original tweet's count (every reshare has ≥ 1), so using reshares would leak the answer.
- **At most 20 tweets per user, and a train/test split by user.** One account (the official AWS feed) holds 43% of all retweets, and the top 1% of users hold 75%. Without the cap, test R² swung between −0.03 and 0.18 depending on which fold that account landed in.
- **`Hour` is US Mountain time for every tweet.** The snowflake TweetIDs confirm this, so hour effects only make sense for US posters.

| Question | Result |
|---|---|
| Q1 mood from text | TF-IDF + LogisticRegression macro-F1 0.81 (5-fold CV) vs Count + MultinomialNB 0.68 vs baseline 0.27. On human-labeled airline tweets it falls to 0.49: the model learned the lexicon. |
| Q2 positive vs negative retweets (Welch t) | p = 2e-11, but Cohen's d = 0.12: real but negligible |
| Q3 mood × weekday (chi-square) | p = 3e-14, Cramér's V = 0.031: negligible |
| Q4 log retweets (LinearRegression) | test R² 0.088 (5-fold 0.094 ± 0.020) vs mean baseline 0. Adding words with Ridge (α = 10) reaches 0.154. |
| Q4 retweeted at all (LogisticRegression) | AUC 0.72 vs 0.52 with mood features only |
| Drivers (drop-one-group ΔR²) | author 0.044 > post format 0.036 > place 0.006 > mood 0.002 > time 0.0005 |
| Bias–variance | Metadata model: train ≈ test error, so high bias. Ridge with words shows the U-shaped validation curve. |
| Bonus (outside course scope) | Fine-tuned DistilBERT on text: AUC 0.76. Laya (open, Jev-compatible): zero-shot airline mood macro-F1 0.69 vs 0.49 for our lexicon-trained classifier; reach AUC 0.50 zero-shot, 0.73 fine-tuned on text, 0.70 with the author card as JSON (one epoch each, RTX 4060). |

### Deliverables

| File | Contents | Status |
|---|---|---|
| `final_analysis.ipynb` | Sections 0–8: cleaning, data switch, EDA, classifier, tests, models, bias–variance, limitations, summary | Done; runs in about 3 min |
| `figures/` | 13 slide figures, numbered by section | Done |
| `results.json` | Every number used on the slides and in the README | Done |
| `make_slides.py` → `MoodReach_Final.pptx` | 7 main slides plus backup, with speaker notes and timings | Draft; needs the team's styling and rehearsal |
| `bonus_llm_finetune.ipynb`, `bonus_laya_finetune.ipynb` | LLM easter eggs | Backup slides only |
| `../README.md` | Answer, results table, reproduce steps | Done |

### Task split (each person presents the slides they own, about 40 s each, leaving 1.5 min for Q&A)

| Person | Owns | Presents | Before Dec 3 |
|---|---|---|---|
| A | Section 0–2: data switch, cleaning decisions, EDA | Slides 2–3 (data, concentration) | Rerun the notebook end to end; be ready for "why cap 20 per user?" and "is Reach leakage?" |
| B | Section 3: sentiment classifier | Slide 4 | Explain macro-F1 vs accuracy and the airline transfer check |
| C | Sections 4–6: tests, engagement models, bias–variance | Slides 5–7 | Explain p-value vs effect size, ΔR² ablation and the Ridge curve |
| D | Slides, README, bonus backup slides | Slide 1 (question) and the closing takeaway; runs the clock | Restyle the deck, rehearse twice to under 3.5 minutes, collect Q&A answers |
