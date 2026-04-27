import os
import gc
import json
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login as hf_login

# --- Config ---
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
CSV_DIR    = "/runpod-volume/csvs"
OUTPUT_DIR = "/runpod-volume/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    hf_login(token=HF_TOKEN)
    print(f"HF_TOKEN loaded (last 4 chars: ...{HF_TOKEN[-4:]})")
else:
    print("WARNING: HF_TOKEN not found in environment")

MODELS = [
    # GPT-2
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    # LLaMA (gated)
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    # Mistral
    "mistralai/Ministral-3B-Instruct-2410",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mixtral-8x7B-v0.1",
    # Qwen
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-72B",
    # OPT
    "facebook/opt-1.3b",
    "facebook/opt-6.7b",
    "facebook/opt-66b",
    # BLOOM
    "bigscience/bloom-1b7",
    "bigscience/bloom-7b1",
]

EXP_FILES = {
    1: os.path.join(CSV_DIR, "exp1_ST.csv"),
    2: os.path.join(CSV_DIR, "exp2_ST.csv"),
    3: os.path.join(CSV_DIR, "exp3_ST.csv"),
}


def load_model(model_id):
    print(f"  Loading {model_id} ...")
    token = HF_TOKEN or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
    )
    model.eval()
    print(f"  Ready.")
    return tokenizer, model


def compute_surprisal(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs[range(len(shift_labels)), shift_labels]
    token_surprisals = (-token_log_probs).tolist()
    tokens = tokenizer.convert_ids_to_tokens(shift_labels.tolist())

    sentence_surprisal = sum(token_surprisals)
    mean_surprisal = sentence_surprisal / len(token_surprisals) if token_surprisals else None

    return {
        "tokens":             tokens,
        "token_surprisals":   token_surprisals,
        "sentence_surprisal": sentence_surprisal,
        "mean_surprisal":     mean_surprisal,
        "num_tokens":         len(token_surprisals),
    }


def process_model(model_id):
    safe_model = model_id.replace("/", "_")

    # Check if all output files already exist (resume support)
    pending = [
        (exp_num, csv_path)
        for exp_num, csv_path in EXP_FILES.items()
        if not os.path.exists(
            os.path.join(OUTPUT_DIR, f"results_exp{exp_num}_{safe_model}.csv")
        )
    ]
    if not pending:
        print(f"  All experiments already done — skipping.")
        return

    try:
        tokenizer, model = load_model(model_id)
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        return

    for exp_num, csv_path in pending:
        print(f"\n  Experiment {exp_num} ...")
        df = pd.read_csv(csv_path, index_col=0)
        sentences = df["sentence"].tolist()

        records = []
        for i, s in enumerate(sentences):
            print(f"    [{i+1}/{len(sentences)}]", end="\r")
            try:
                r = compute_surprisal(s, tokenizer, model)
                records.append({
                    "tokens":             json.dumps(r["tokens"]),
                    "token_surprisals":   json.dumps(r["token_surprisals"]),
                    "sentence_surprisal": r["sentence_surprisal"],
                    "mean_surprisal":     r["mean_surprisal"],
                    "num_tokens":         r["num_tokens"],
                    "error":              None,
                })
            except Exception as e:
                records.append({
                    "tokens": None, "token_surprisals": None,
                    "sentence_surprisal": None, "mean_surprisal": None,
                    "num_tokens": None, "error": str(e),
                })

        out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
        out_path = os.path.join(OUTPUT_DIR, f"results_exp{exp_num}_{safe_model}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for model_id in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_id}")
        print(f"{'='*60}")
        process_model(model_id)

    print("\nAll done.")
