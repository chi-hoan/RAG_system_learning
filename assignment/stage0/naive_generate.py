"""
Stage 0B — Read the loop: generation from scratch, no `.generate()`.

The model is a function: (token_ids) -> logits. Generation is a loop *we* write
around that function. This file implements that loop by hand to make the
mechanics impossible to hand-wave.

Run:
    python naive_generate.py

Requires: torch, transformers. Loads Qwen/Qwen3-0.6B on first run (~1.2 GB).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The three most important ideas in machine learning are"
N_NEW_TOKENS = 30
TEMPERATURE = 0.8  # try 0.01 and 5.0 — see comment (2) at the bottom


def naive_generate(model, tokenizer, prompt, n_new_tokens, temperature):
    """Autoregressive sampling loop. No KV cache, no `.generate()`."""
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    for _ in range(n_new_tokens):
        with torch.no_grad():
            # Forward pass over the ENTIRE sequence every step. The model
            # returns logits of shape (batch, seq_len, vocab_size) — one
            # distribution per position. See comment (1): we use exactly one.
            outputs = model(input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Take the logits at the LAST position only — the prediction for the
        # next token. Every other position is discarded.
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

        # Temperature scales the logits BEFORE softmax. Dividing by a small
        # number sharpens the distribution; dividing by a large one flattens it.
        scaled = next_token_logits / temperature

        # Softmax turns logits into a probability distribution.
        probs = torch.softmax(scaled, dim=-1)  # (1, vocab_size)

        # Sample one token from that distribution (stochastic — not argmax).
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Append and repeat. The sequence grows by one token each iteration.
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    print(f"Loading {MODEL_ID} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
    ).to(device)
    model.eval()

    text = naive_generate(model, tokenizer, PROMPT, N_NEW_TOKENS, TEMPERATURE)
    print("\n=== Output ===")
    print(text)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# ANSWERS
# ─────────────────────────────────────────────────────────────────────────────
#
# (1) The forward pass returned logits for *every* position. How many did you
#     use? What's the waste?
#
#     Out of the seq_len distributions returned each step, I used exactly ONE:
#     the last position, `logits[:, -1, :]`. Every earlier position's logits
#     were thrown away.
#
#     But the waste is deeper than the discarded logits. Because the loop feeds
#     the WHOLE growing sequence back in every iteration, the model recomputes
#     the attention keys and values for all previous tokens on every single
#     step — even though those tokens haven't changed. Step t re-derives the
#     same K/V it derived at step t-1, t-2, ... This makes generation O(n^2) in
#     sequence length instead of O(n).
#
#     The fix is the KV CACHE: store each token's key/value vectors the first
#     time they're computed and reuse them, so each new step only runs the
#     forward pass on the ONE new token. That is exactly what `use_cache=True`
#     (and real `.generate()`) does, and why removing it is the single biggest
#     inference slowdown at long context.
#
# (2) Set temperature to 0.01 and to 5.0. Explain in terms of the softmax
#     denominator.
#
#     Softmax is exp(z_i / T) / Σ_j exp(z_j / T). The denominator is the sum
#     over the whole vocabulary of the exponentiated, temperature-scaled logits.
#
#     T = 0.01: dividing by 0.01 multiplies every logit gap by ~100. After exp,
#     the single largest logit utterly dominates the denominator — its term is
#     astronomically larger than all others combined. The distribution collapses
#     onto the top token (probability ≈ 1). Output becomes near-greedy:
#     deterministic, repetitive, "safe."
#
#     T = 5.0: dividing by 5 shrinks all the gaps toward zero. After exp, every
#     term is close to exp(0)=1, so the denominator is a sum of many comparable
#     values and no single token dominates. The distribution flattens toward
#     uniform over the 150K-token vocab, so multinomial routinely samples from
#     the low-probability tail — output becomes incoherent, "random" garbage.
#
# (3) Remove the temperature division entirely. What temperature is that
#     equivalent to?
#
#     Removing `/ temperature` is the same as dividing by 1.0. So plain
#     softmax(logits) == temperature 1.0 — the model's own, unmodified
#     distribution. Temperature is not part of the model; it is a sampling
#     knob applied to the logits after the fact, and T=1.0 is the identity.
