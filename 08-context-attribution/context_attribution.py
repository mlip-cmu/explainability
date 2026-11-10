"""Context attribution: which sentences of the context did an LLM answer depend on?"""

import warnings

import numpy as np
import torch
from sklearn.linear_model import Lasso
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
logging.disable_progress_bar()
torch.manual_seed(0)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # a small open model: 1 GB, runs on a CPU
REPORT = [
    "The defendant is 24 years old and lives in the south district of the county.",
    "He was arrested on 3 March for possession of a controlled substance, a felony.",
    "He has five prior convictions, the last one in 2023 for theft.",
    "He has no juvenile record.",
    "At the time of the arrest he was unemployed.",
    "He lost his job at a warehouse in January.",
    "He is not enrolled in a drug treatment program.",
    "His sister has offered him a room in her apartment.",
    "He appeared at all eleven court dates in his earlier cases.",
    "Forum post, unverified: people like him never show up in court, he missed three hearings last year.",
    "The public defender asks for release without bail.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).eval()


def prompt(sentences: list[str], question: str) -> torch.Tensor:
    text = (
        "Pretrial report:\n"
        + " ".join(sentences)
        + f"\n\nQuestion: {question} Answer in one sentence."
    )
    chat = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]


@torch.no_grad()
def answer(sentences: list[str], question: str, max_new_tokens: int = 40) -> str:
    ids = prompt(sentences, question)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


@torch.no_grad()
def log_probability(sentences: list[str], question: str, text: str) -> float:
    """How likely is this exact answer, if the model sees only these sentences?"""
    ids = prompt(sentences, question)
    target = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    logits = model(torch.cat([ids, target], dim=1)).logits[0, ids.shape[1] - 1 : -1]
    return float(torch.log_softmax(logits, dim=-1).gather(1, target[0][:, None]).sum())


def attribute(question: str, n_samples: int = 64, seed: int = 0):
    """ContextCite: remove random subsets of the sentences, then fit a sparse linear model."""
    original = answer(REPORT, question)
    rng = np.random.default_rng(seed)
    masks = rng.random((n_samples, len(REPORT))) < 0.5  # True: the sentence stays
    kept = [[s for s, keep in zip(REPORT, mask) if keep] for mask in masks]
    scores = np.array([log_probability(sentences, question, original) for sentences in kept])
    surrogate = Lasso(alpha=0.25).fit(masks.astype(float), scores)
    return original, surrogate.coef_, surrogate.score(masks.astype(float), scores)


def show(weights: np.ndarray, top: int = 4) -> None:
    for i in np.argsort(-np.abs(weights))[:top]:
        if weights[i] != 0:
            print(f"   {weights[i]:+6.2f}  [{i + 1:2d}] {REPORT[i]}")
    print(f"   ({int(np.sum(weights == 0))} of the {len(weights)} sentences have the weight 0)")


print("1. A question about a pretrial report with 11 sentences")
question = "Did the defendant miss court dates in the past?"
for i, sentence in enumerate(REPORT, 1):
    print(f"   [{i:2d}] {sentence}")
original, weights, fit = attribute(question)
print(f"   Question: {question}")
print(f"   Answer of the model: {original}")
print("   Sentence 9 (the court record) says the opposite. Where does the answer come from?")

print("\n2. Ask the model for its source")
source = answer(REPORT, f"{question} Which sentence of the report is your source? Quote it.", 60)
print(f"   {source}")
print("   A citation is generated text too. Nothing ties it to what the answer depended on.")

print("\n3. Context attribution: 64 random subsets of the sentences, one weight per sentence")
print("   weight = change of the log-probability of the answer when the sentence is in the context")
show(weights)
print(f"   R2 of the linear model on the 64 samples: {fit:.2f}")

print("\n4. Check the attribution: remove one sentence and ask again")
top = int(np.argmax(np.abs(weights)))
low = int(np.argmin(np.abs(weights)))
before = log_probability(REPORT, question, original)
for label, i in [
    ("the sentence with the largest weight", top),
    ("a sentence with the smallest weight", low),
]:
    rest = [s for j, s in enumerate(REPORT) if j != i]
    change = log_probability(rest, question, original) - before
    print(f"   without [{i + 1}], {label}:")
    print(f"      log-probability of the first answer changes by {change:+.1f}")
    print(f"      new answer: {answer(rest, question)}")

print("\n5. A second question: the weights are different for each answer")
question = "How many prior convictions does the defendant have?"
original, weights, fit = attribute(question)
print(f"   Question: {question}")
print(f"   Answer of the model: {original}")
show(weights, top=3)
