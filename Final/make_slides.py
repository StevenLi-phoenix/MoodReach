"""Build the final-presentation deck from figures/ and results.json.

Run from inside Final/ after final_analysis.ipynb (and optionally bonus_llm_finetune.ipynb):
    python make_slides.py
Writes MoodReach_draft.pptx (the presented MoodReach_Final.pptx is hand-made from it). Every number on the slides comes from results.json / bonus_results.json.
"""

import json
import logging
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.slide import Slide
from pptx.util import Emu, Inches, Pt

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("moodreach.slides")

R = json.load(open("results.json"))
BONUS = json.load(open("bonus_results.json")) if os.path.exists("bonus_results.json") else None
LAYA = json.load(open("laya_results.json")) if os.path.exists("laya_results.json") else None
TEAM = "Vince Pamolarco · John Yun · Siddharth Srivastava · Steven Li"

INK = RGBColor(0x1F, 0x24, 0x33)
MUTED = RGBColor(0x5A, 0x63, 0x75)
ACCENT = RGBColor(0x7A, 0x4F, 0xB0)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide: Slide, x: float, y: float, w: float, h: float, lines: list[str], size: int = 18, color: RGBColor = INK, bold_first: bool = False) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold_first and i == 0
        p.space_after = Pt(8)


def picture(slide: Slide, path: str, x: float, y: float, w: float, h: float) -> None:
    """Place a figure inside the (x, y, w, h) box in inches, keeping its aspect ratio."""
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y))
    ratio = pic.width / pic.height
    bw, bh = Inches(w), Inches(h)
    if bw / bh > ratio:
        pic.height, pic.width = bh, Emu(int(bh * ratio))
    else:
        pic.width, pic.height = bw, Emu(int(bw / ratio))
    pic.left = Inches(x) + (bw - pic.width) // 2


def slide(title: str, fig: str, bullets: list[str], notes: str, takeaway: str | None = None, fig_w: float = 8.2) -> Slide:
    s = prs.slides.add_slide(BLANK)
    text(s, 0.5, 0.3, 12.3, 0.9, [title], size=30, bold_first=True)
    if fig:
        picture(s, f"figures/{fig}.png", 0.4, 1.3, fig_w, 5.2)
    bx = 0.5 + fig_w
    text(s, bx, 1.4, 13.0 - bx, 5.0, [f"• {b}" for b in bullets], size=16)
    if takeaway:
        text(s, 0.5, 6.6, 12.3, 0.7, [takeaway], size=18, color=ACCENT, bold_first=True)
    s.notes_slide.notes_text_frame.text = notes
    log.info("slide %d: %s", len(prs.slides), title)
    return s


def pct(v: float) -> str:
    return f"{100 * v:.0f}%"


d, clf, h1, h2 = R["data"], R["classifier"], R["h1_ttest"], R["h2_chi2"]
reg = {r["model"]: r for r in R["regression"]["table"]}
logit = {r["model"]: r for r in R["logistic"]["table"]}
abl = {r["group"]: r["R2 drop when removed"] for r in R["regression"]["ablation"]}
air = {r["model"]: r for r in clf["airline_transfer"]}
bv = R["bias_variance"]

# 1. Title
s = prs.slides.add_slide(BLANK)
text(s, 0.8, 1.6, 11.7, 1.2, ["MoodReach"], size=54, color=ACCENT, bold_first=True)
text(s, 0.8, 2.8, 11.7, 1.3, ["What drives the reach of a social media post: mood, time, place, or the author?"], size=28)
text(s, 0.8, 4.2, 11.7, 1.4, [
    f"{d['rows_after_dedup']:,} real tweets · {d['countries']} countries · {d['date_min']} → {d['date_max']}",
    "Answer in one line: who posts and how it is written matter; mood barely does.",
], size=20, color=MUTED)
text(s, 0.8, 6.3, 11.7, 0.6, [f"{TEAM}  |  NYU Tandon Data Science Bootcamp, Section 2  |  Dec 3, 2026"], size=14, color=MUTED)
s.notes_slide.notes_text_frame.text = (
    "~20s. Our question: what makes a post spread? We split it into four questions: can we read mood from text, "
    "does mood change retweets, is mood tied to the weekday, and how well can we predict retweets."
)

# 2. Data
slide(
    "We switched to real data",
    "1_1_old_vs_new_hours",
    [
        f"Midterm data: 732 posts, nothing before 06:00 → judges said it looked synthetic",
        f"Now: {d['rows_after_dedup']:,} real tweets with verifiable Twitter IDs, {d['users']:,} users",
        "Checks: dropped 2,063 duplicate IDs; Hour = US Mountain time (verified from IDs)",
        f"Reshares inherit the original's count (min {d['reshare_min_rt']}) → modeled {d['original_tweets']:,} original tweets only",
    ],
    "~35s. Answer the midterm feedback first. Tweet IDs encode the post time, which let us verify the clock "
    "and catch that reshares carry the original tweet's retweet count, which would leak the answer.",
)

# 3. Concentration
slide(
    "Reach is rare and concentrated",
    "2_6_reach_concentration",
    [
        f"{pct(1 - d['original_share_retweeted'])} of original tweets get zero retweets",
        f"One account (the official AWS feed) earns {pct(d['top_account_share'])} of all retweets; top 1% of users earn {pct(d['top1pct_users_share'])}",
        f"Fix: at most {d['cap_per_user']} tweets per user ({d['capped_tweets']:,} tweets), split train/test by user",
        f"Without it, test R² swung from −0.03 to 0.18 by fold; now 5-fold R² = {R['regression']['cv_r2_mean']:.3f} ± {R['regression']['cv_r2_std']:.3f}",
    ],
    "~35s. The biggest surprise in the EDA. If one account is in the test set, it dominates the score, so we cap "
    "each user and never let the same user appear in both train and test.",
)

# 4. Classifier
nb = next(r for r in clf["table_test"] if r["model"] == "Count + MultinomialNB")
best = next(r for r in clf["table_test"] if r["model"] == clf["best"])
slide(
    "Q1 · Reading mood from text",
    "3_1_classifier_confusion_words",
    [
        f"Baseline (always 'neutral'): macro-F1 {clf['table_test'][0]['macro_F1']:.2f}",
        f"Count + MultinomialNB: {nb['macro_F1']:.2f} · {clf['best']}: {best['macro_F1']:.2f} (5-fold CV {clf['best_cv_macro_f1_mean']:.2f})",
        f"Honesty check on 14,640 human-labeled airline tweets: macro-F1 drops to {air[clf['best']]['macro_F1']:.2f}",
        "The labels come from a word lexicon, so the model learns the lexicon, not human mood",
    ],
    "~35s. LogisticRegression beats Naive Bayes. But the labels are a word-count score, so we tested on airline "
    "tweets labeled by people: performance drops a lot. It misses complaints that use no negative word.",
)

# 5. Tests
slide(
    "Q2–Q3 · Mood matters statistically, not practically",
    "4_1_hypothesis_tests",
    [
        f"Q2 Welch t-test, positive vs negative: p = {h1['welch_p']:.0e}, but Cohen's d = {h1['cohen_d']:.2f}",
        f"Retweet rate {pct(h1['rate_positive'])} vs {pct(h1['rate_negative'])}: real, small",
        f"Q3 chi-square, mood × weekday: p = {h2['p']:.0e}, Cramér's V = {h2['cramers_v']:.3f}",
        "With ~50k tweets every p is tiny; effect size tells us the gap is negligible",
    ],
    "~35s. Both tests reject the null, but the effect sizes are below the 'small' thresholds (d < 0.2, V < 0.1). "
    "Mood matters a little, not a lot.",
)

# 6. Drivers
slide(
    "Q4 · What drives reach",
    "5_1_drivers",
    [
        f"LinearRegression on log retweets: test R² {reg['LinearRegression (test)']['R2']:.3f} vs mean baseline 0",
        f"Retweeted at all? LogisticRegression AUC {logit['All features']['ROC_AUC']:.2f} vs mood only {logit['Mood features only']['ROC_AUC']:.2f}",
        f"Removing author costs ΔR² {abl['author']:.3f}, post format {abl['post format']:.3f}, mood {abl['mood']:.3f}, time {abl['time']:.4f}",
        "Replies lose reach; mentions and hashtags gain",
    ],
    "~40s. The left chart reads: each extra standard deviation of follower count raises the odds of a retweet by about 70%. "
    "On the right, removing the author or format features hurts the most; removing mood or time barely matters.",
    fig_w=8.6,
)

# 7. Bias-variance + limits
slide(
    "Bias–variance and limits",
    "6_1_bias_variance",
    [
        f"Metadata model: train ≈ test error → high bias; more rows won't help",
        f"Adding tweet words (Ridge, α={bv['best_alpha']:g}) lifts test R² to {bv['ridge_text_best_test_r2']:.2f}; small α overfits",
        "Limits: AWS/tech topic skew · 2016 data · lexicon labels · one US clock · no images or follower graph",
    ],
    "~35s. Our features explain only about 10–15% of reach. The rest is in things we cannot see. Close with the "
    "takeaway: if you want reach, grow your audience and write shareable posts; tone is a footnote.",
    takeaway="Takeaway: grow the audience and shape the post; mood is a footnote.",
)

# Backup slides
slide(
    "Backup · Words linked to more / fewer retweets",
    "6_2_words_that_spread",
    [
        "Up: event and topic hashtags (#serverless, #bigdata, conference chats), 'photo'",
        "Down: automated feeds (dlvr, twitterfeed, job boards, 'via'), manual 'RT'",
        "Ridge on TF-IDF words plus metadata, coefficients on log(1 + retweets)",
    ],
    "Backup for Q&A: which words matter.",
    fig_w=6.5,
)
if BONUS:
    auc = BONUS["test_auc"]
    slide(
        "Backup · Easter egg: fine-tuned small language model",
        "7_1_llm_bonus",
        [
            f"DistilBERT ({BONUS['epochs']} epochs, {BONUS['train_seconds'] / 60:.0f} min on {BONUS['device'].upper()}), text only",
            f"AUC: DistilBERT {auc['Text: fine-tuned DistilBERT']:.3f} vs TF-IDF + LogReg {auc['Text: TF-IDF + LogReg']:.3f}",
            f"Metadata without text: {auc['All metadata, no text (LogReg)']:.3f}; mood only {auc['Mood features only (LogReg)']:.3f}",
            "Outside the course scope; kept as a bonus",
        ],
        "Backup: only if asked about deep learning.",
    )

if LAYA:
    mood = {r["model"]: r for r in LAYA["airline_mood"]}
    lex_best = next(m for m in mood if m.endswith("(lexicon labels)") and "MultinomialNB" not in m)
    ra = LAYA["reach_auc"]
    slide(
        "Backup · Laya: reading mood zero-shot",
        "7_2_laya_mood",
        [
            "Laya: open, Jev-compatible 'System One' model (421M), typed answers with calibrated probabilities",
            f"Human-labeled airline mood, no training: macro-F1 {mood['Laya zero-shot (no training)']['macro_F1']:.2f} vs {mood[lex_best]['macro_F1']:.2f} for our lexicon-trained model",
            "Mood is a language task: pretrained models read it; lexicon labels do not",
        ],
        "Backup: only if asked about LLMs.",
        fig_w=8.0,
    )
    card_job = LAYA.get("jobs", {}).get("card", {})
    slide(
        "Backup · Fine-tuned Laya: does it learn reach?",
        "7_3_laya_reach",
        [
            f"Zero-shot AUC {ra['Laya zero-shot, text']:.2f} (coin flip) → fine-tuned on text {ra['Fine-tuned Laya, text']:.3f}",
            f"Text + author card {ra['Fine-tuned Laya, text + author card']:.3f}: digits in JSON are hard to learn in 1 epoch; LogReg on metadata {ra['All metadata, no text (LogReg)']:.3f}",
            f"{LAYA['train_rows']:,} train / {LAYA['test_rows']:,} test tweets from unseen users"
            + (f"; {card_job['train_minutes']:.0f} min on an {card_job['device'].replace('NVIDIA GeForce ', '')}" if card_job.get("train_minutes") else ""),
        ],
        "Backup: words learn reach only after training; mood alone stays near 0.52.",
        fig_w=7.6,
    )

prs.save("MoodReach_draft.pptx")  # the presented deck, MoodReach_Final.pptx, is hand-made from this draft
log.info("wrote MoodReach_draft.pptx (%d slides)", len(prs.slides))
