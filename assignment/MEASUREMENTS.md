# MEASUREMENTS.md

Every number and every triage decision, dated. The most valuable file in the repo.

The five levers (roadmap §1): **Context** (prompt, RAG, examples — free, instant), **Sampling** (temperature, top-p — free, instant), **Precision** (quantization — minutes, reload), **Thinking budget** (reasoning tokens — linear in tokens, instant), **Weights** (SFT/LoRA/DPO — weeks, retrain). Rule: name the lever *before* you touch anything.

---

## Lever triage

Ten complaints about an LLM system. For each: the lever, the cheapest intervention that could fix it, and what it costs you if you pulled the wrong lever instead.

| # | Complaint | Lever | Cheapest intervention | Cost if you chose wrong |
|---|-----------|-------|-----------------------|-------------------------|
| 1 | "It makes up article numbers." | **Context** (+ Stage 4 scoring) | Retrieve the actual article and require a citation; score abstention so "I don't know" beats a confident guess. | Fine-tune on the statute → *weeks*, degrades other skills, still fabricates when the law changes. The classic trap. |
| 2 | "It doesn't know our 2026 policy." | **Context** | Put the policy in the prompt / RAG index. A fact that changes cannot live in the weights. | Fine-tune facts in → confidently wrong the day the policy updates, and you re-train every revision. |
| 3 | "It won't follow my JSON format." | **Sampling** | Constrained decoding / grammar (JSON mode) + schema and one example in the prompt. | Fine-tune for format → weeks of work for what a decoding constraint fixes in an afternoon. |
| 4 | "Answers are too short." | **Context** | Raise `max_tokens` and instruct "be thorough, show reasoning." | Assume it's a capability ceiling and swap/retrain the model — you never tested the free instruction. |
| 5 | "Same question, different answer every time." | **Sampling** | Set `temperature = 0` (greedy) for deterministic tasks. | Chase it as a precision/weights bug for days; it was one config field. |
| 6 | "Occasionally it emits something insane." | **Sampling** | Truncate the tail: `top_p ≈ 0.9` / `min_p`, `top_k`. The bottom of a 150K vocab holds nonzero garbage mass. | Blame the weights and retrain, or over-lower temperature and make it dull, without fixing the tail. |
| 7 | "It's too slow." | **Thinking budget** (+ streaming) | Cap reasoning tokens / `max_tokens`; stream so TTFT drops from 20s to ~1.5s felt. | Quantize below Q4 to "speed it up" → silent, undetectable degradation of reasoning and code (roadmap §1.3). |
| 8 | "It refuses reasonable requests." | **Context** | Rewrite the system prompt; add in-scope examples of what to answer. | DPO/RLHF the refusals away → weeks, and you can dial the same behavior with a sentence. |
| 9 | "It got measurably worse at code after we shrank it to fit VRAM." | **Precision** | Move Q3/Q4 → Q8 (≈ fp16) or a better-calibrated quant (AWQ); reload. | Assume the base model is bad and start fine-tuning, when the loss was in the storage format. |
| 10 | "It won't adopt our house terminology, even with 5 examples in every prompt." | **Weights** (LoRA) | *After* Stage 4 proves prompting/RAG genuinely can't hold the style: a small LoRA on curated in-house text. | Keep bloating the prompt with examples forever — cost per call, latency, and it still drifts. This is one of the rare *real* weights cases. |

**Distribution:** Context ×4 (1,2,4,8), Sampling ×3 (3,5,6), Thinking budget ×1 (7), Precision ×1 (9), Weights ×1 (10).
**Pass check:** 7/10 resolve to **context** or **sampling** (≥7 required); only **1** to weights (≤2 required). ✔

### Notes on the two that hurt most

- **Row 1 is the trap.** "It hallucinates → the weights are wrong" is the field's most expensive instinct. It's almost always a *retrieval* failure (the right fact was never in context) compounded by a *scoring* failure (Stage 4: the industry rewards guessing over abstaining). Fix both cheaply before you ever touch a weight.
- **Row 10 is the one legitimate weights case**, and it's still gated: only pull it once your Stage 4 eval set shows that context and RAG genuinely cannot carry the behavior. Persistent *style/behavior* that survives good prompting is what weights are for — persistent *facts* never are (that's row 2).
