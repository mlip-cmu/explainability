# 08 · Context attribution for LLM answers

A court clerk uses an LLM to answer questions about a pretrial report. The report in this
project has 11 sentences about a defendant. One of them is not part of the official record: an
unverified forum post that says that he missed three hearings.

**Problem.** The LLM answers "Yes, he missed court dates", although the court record in the
report says that he appeared at all eleven. Which part of the context did the answer depend on?
You can ask the model for its source, but a citation is generated text too, and nothing ties
it to the computation that produced the answer.

**Idea.** The same idea as LIME, with the sentences of the context as the features
([ContextCite](https://arxiv.org/abs/2409.00729)). Remove random subsets of the sentences,
measure for each subset how probable the original answer still is, and fit a sparse linear
model: one weight per sentence. A sentence with a large weight is one that the answer depends
on. The method needs the probabilities of the model, not its internals.

How probable is the original answer when the model sees only some of the sentences
(`context_attribution.py`):

```python
ids = prompt(sentences, question)
target = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
logits = model(torch.cat([ids, target], dim=1)).logits[0, ids.shape[1] - 1 : -1]
return float(torch.log_softmax(logits, dim=-1).gather(1, target[0][:, None]).sum())
```

64 random subsets, then one weight per sentence:

```python
masks = rng.random((n_samples, len(REPORT))) < 0.5  # True: the sentence stays
kept = [[s for s, keep in zip(REPORT, mask) if keep] for mask in masks]
scores = np.array([log_probability(sentences, question, original) for sentences in kept])
surrogate = Lasso(alpha=0.25).fit(masks.astype(float), scores)
```

## What the code shows

`context_attribution.py`:

1. The report, the question "Did the defendant miss court dates in the past?", and the answer
   of the model: yes, "he had missed three hearings last year".
2. **The model's own citation is wrong.** Asked for its source, the model quotes the sentence
   "At the time of the arrest he was unemployed".
3. **The attribution is clear.** The forum post has the weight +27.8; the next sentence has
   +0.7, and 7 of the 11 sentences have the weight 0. The linear model fits the 64 samples
   with R² = 0.97.
4. **Check.** Without the forum post, the log-probability of the first answer decreases by
   28.2, and the three hearings are not in the new answer. Without a sentence of weight 0, it
   changes by 0.1. But the model still answers "yes" without the forum post, and gives a
   different reason: attribution explains an answer, it does not make the model correct.
5. A second question ("How many prior convictions?") has a different attribution: only the
   sentence about the convictions has a weight (+18.7).

## Tools

- [Transformers](https://huggingface.co/docs/transformers): a library to load and run open
  language models. Here: it runs the model and gives the probability of each token.
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): a small open
  chat model (0.5 billion parameters, 1 GB). Here: the LLM that is explained. It is small enough
  for a laptop CPU; it also makes more errors than a large model.
- [PyTorch](https://pytorch.org): the tensor library below Transformers (the CPU build).
- [scikit-learn](https://scikit-learn.org): `Lasso`, the sparse linear model.
- The [context-cite](https://github.com/MadryLab/context-cite) library has the complete method;
  this project implements the idea in about 40 lines.

## Run

With [uv](https://docs.astral.sh/uv/). The first run downloads the model (1 GB); a run takes
about 30 seconds on four CPU cores.

```sh
uv run context_attribution.py
```
