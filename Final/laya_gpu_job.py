"""GPU job for bonus_laya_finetune.ipynb: zero-shot and fine-tuned Laya (open, Jev-compatible decision model).

The notebook writes the inputs to data/laya_job/; run this wherever a GPU is free, then copy *.csv / meta_*.json back:
    python laya_gpu_job.py --work data/laya_job --device cuda   # ~3.7 h on an RTX 4060 laptop GPU (two 1-epoch fine-tunes)
Every step writes its own output and is skipped when that output exists, so the job can resume after a crash.
Needs: torch, pandas, laya (>= 0.5, `uv pip install laya`).
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")  # laya docs: TensorFlow probing can deadlock model construction

import laya
import pandas as pd
import torch
from laya.train import TrainConfig, finetune

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s", force=True)
log = logging.getLogger("moodreach.laya.job")
BASE = "convaiinnovations/laya"


def free(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def predict(agent, states: list, questions: dict, qid: str, key: str) -> list:
    t = time.time()
    res = agent.predict_batch(states, questions, batch_size=64, sort_by_length=True)
    log.info("predicted %d states in %.0fs", len(states), time.time() - t)
    return [r["answers"][qid][key] for r in res]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="data/laya_job")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--micro-batch", type=int, default=4, help="4 fits an 8 GB GPU with AMP + gradient checkpointing")
    args = ap.parse_args()
    W, dev = args.work, args.device
    q = json.load(open(f"{W}/questions.json"))
    test = pd.read_csv(f"{W}/test.csv")
    card_states = [json.loads(s) for s in test.state_card]
    text_states = [json.loads(s) for s in test.state_text]
    log.info("device %s, torch %s, laya %s, test rows %d", dev, torch.__version__, getattr(laya, "__version__", "?"), len(test))
    if dev == "cuda":
        log.info("GPU %s, free/total bytes %s", torch.cuda.get_device_name(0), torch.cuda.mem_get_info())

    # 1. zero-shot: airline mood + reach (both variants)
    if not os.path.exists(f"{W}/pred_zero_shot.csv"):
        base = laya.load(BASE, device=dev)
        air = pd.read_csv(f"{W}/airline.csv")
        air["laya_zero_shot"] = predict(base, air.text.tolist(), q["sentiment"], "sentiment", "choice")
        air[["tweet_id", "airline_sentiment", "laya_zero_shot"]].to_csv(f"{W}/pred_airline.csv", index=False)
        out = test[["TweetID", "retweeted"]].copy()
        out["p_zero_card"] = predict(base, card_states, q["reach_card"], "retweeted", "noul")
        out["p_zero_text"] = predict(base, text_states, q["reach_text"], "retweeted", "noul")
        out.to_csv(f"{W}/pred_zero_shot.csv", index=False)
        del base
        free(dev)

    # 2. fine-tune each variant, then score the held-out users
    for variant, states, max_len in [("text", text_states, 96), ("card", card_states, 128)]:
        pred_path = f"{W}/pred_finetuned_{variant}.csv"
        if os.path.exists(pred_path):
            log.info("skip %s (done)", variant)
            continue
        ckpt = f"{W}/ckpt_{variant}"
        t0 = time.time()
        if not os.path.exists(f"{ckpt}/model.safetensors"):
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            report = finetune(
                f"{W}/train_{variant}.jsonl", BASE, ckpt,
                TrainConfig(epochs=1, micro_batch=args.micro_batch, grad_accum=64 // args.micro_batch, max_len=max_len, head_max_len=64,
                            log_every=200, seed=42, amp=dev == "cuda", gradient_checkpointing=dev == "cuda"),
                device=dev,
            )
            meta = {
                "variant": variant,
                "device": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
                "train_minutes": round((time.time() - t0) / 60, 1),
                "train_items": report.get("train_items"),
                "epoch_loss": report.get("epoch_loss"),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None,
            }
            json.dump(meta, open(f"{W}/meta_{variant}.json", "w"), indent=2, default=str)
            log.info("fine-tuned %s: %s", variant, meta)
            free(dev)
        tuned = laya.load(ckpt, device=dev)
        out = test[["TweetID", "retweeted"]].copy()
        out[f"p_finetuned_{variant}"] = predict(tuned, states, q[f"reach_{variant}"], "retweeted", "noul")
        out.to_csv(pred_path, index=False)
        del tuned
        free(dev)
    log.info("all done")


if __name__ == "__main__":
    main()
