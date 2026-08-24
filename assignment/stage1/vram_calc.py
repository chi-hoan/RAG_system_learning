"""
Stage 1B — VRAM and bandwidth calculator: predict fit and speed from a model
card, before downloading 16GB.

Given (params, quant, n_layers, n_kv_heads, head_dim, ctx) it returns weights
GB, KV cache GB, total GB, and the theoretical tok/s ceiling (memory-bandwidth
bound). The architecture fields are pulled straight from each model's
config.json on the Hub, so the inputs are real, not guessed.

Run:
    python vram_calc.py

Requires: transformers (for AutoConfig). Reading a config.json is a tiny
download; the weights are never fetched.
"""

from transformers import AutoConfig

# ── The four core formulas (as given in the assignment) ──────────────────────

def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_elem=2):
    # 2 = one Key + one Value per layer. bytes_per_elem stays 2 (fp16) even for
    # a quantized model: the KV cache is almost always kept in fp16, regardless
    # of how the WEIGHTS are stored. Quantizing weights does not shrink the KV.
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem


def vram_estimate_gb(params_b, bytes_per_param, n_layers, n_kv_heads,
                     head_dim, ctx_tokens, overhead_gb=1.0):
    weights = params_b * 1e9 * bytes_per_param
    kv = kv_bytes_per_token(n_layers, n_kv_heads, head_dim) * ctx_tokens
    return (weights + kv) / 1e9 + overhead_gb


def weights_gb(params_b, bytes_per_param):
    return params_b * 1e9 * bytes_per_param / 1e9


def kv_gb(n_layers, n_kv_heads, head_dim, ctx_tokens):
    return kv_bytes_per_token(n_layers, n_kv_heads, head_dim) * ctx_tokens / 1e9


def tps_ceiling(bandwidth_gbs, active_model_gb):
    # Decoding is memory-bandwidth bound: each token requires reading the whole
    # (active) model from VRAM once. Ceiling = bandwidth / bytes-read-per-token.
    return bandwidth_gbs / active_model_gb


# ── bytes/param per quantization; KV always fp16 (see above) ──────────────────
QUANT_BYTES = {"fp16": 2.0, "bf16": 2.0, "q8": 1.0, "q4": 0.5}

# Total params (from each model card) — AutoConfig doesn't expose a param count.
MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", 8.03),
    ("Qwen/Qwen3-8B", 8.19),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", 23.57),
]

# tok/s ceiling depends only on hardware bandwidth + active weights size.
HARDWARE = {
    "RTX 4090 (1008 GB/s)": 1008,
    "RTX 3090 (936 GB/s)": 936,
    "Mini PC DDR5 (~40 GB/s)": 40,
}

QUANT = "q4"
CTX = 32_768
# CUDA context + activations + framework. The assignment's default 1.0 GB is
# too low in practice (you'll read "consistently low"); ~2.0 GB is closer for
# a single 8-24B model with a real runtime. Tune against nvidia-smi and say so.
OVERHEAD_GB = 2.0


def arch_from_config(cfg):
    """Pull n_layers / n_kv_heads / head_dim, handling multimodal wrappers
    (Gemma-3, Mistral-Small vision variants nest the LM under text_config)."""
    if not hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return n_layers, n_kv_heads, head_dim


def main():
    bpp = QUANT_BYTES[QUANT]
    print(f"quant={QUANT} ({bpp} bytes/param)   ctx={CTX:,}   overhead={OVERHEAD_GB} GB\n")
    print(f"{'Model':<48} {'W GB':>6} {'KV GB':>6} {'Total':>6}")
    print("-" * 70)

    for model_id, params_b in MODELS:
        try:
            cfg = AutoConfig.from_pretrained(model_id)
        except Exception as e:
            print(f"{model_id:<48}   SKIPPED — {type(e).__name__}: {str(e).splitlines()[0][:50]}")
            continue

        n_layers, n_kv_heads, head_dim = arch_from_config(cfg)
        w = weights_gb(params_b, bpp)
        kv = kv_gb(n_layers, n_kv_heads, head_dim, CTX)
        total = vram_estimate_gb(params_b, bpp, n_layers, n_kv_heads, head_dim, CTX, OVERHEAD_GB)
        print(f"{model_id:<48} {w:>6.2f} {kv:>6.2f} {total:>6.2f}"
              f"   (L={n_layers}, kv_heads={n_kv_heads}, head_dim={head_dim})")

        for hw, bw in HARDWARE.items():
            print(f"    {hw:<28} ceiling ≈ {tps_ceiling(bw, w):>6.1f} tok/s")
        print()


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────────────────────────────────────
#
# WEIGHTS vs KV: weights are fixed; the KV cache grows linearly with
# ctx_tokens × concurrency. Short context OOMs never happen; long context ones
# do. That's the whole cost structure of long context in one line.
#
# KV stays fp16 even at Q4: quantizing weights halves/quarters the weight term
# but does nothing to the KV term. At very long context the KV can exceed the
# weights — which is why paged/quantized KV and GQA (few kv_heads) exist.
#
# tok/s ceiling is memory-bound, not compute-bound: one token = read the active
# weights once. bandwidth ÷ weights_gb. A 4.5 GB model on 1008 GB/s → 224 tok/s;
# the same model on a 40 GB/s mini PC → ~9 tok/s. Same model, 25x slower — the
# bottleneck is the bus, not the FLOPs.
#
# Predicted vs nvidia-smi: two known gap sources — (1) GB here is decimal (1e9)
# but nvidia-smi reports MiB (1024-based), ~7% apparent inflation; (2) OVERHEAD
# must cover CUDA context + activations + framework. If you're CONSISTENTLY low,
# raise OVERHEAD_GB and note it — that's the assignment's expected correction.
#
# Fill the printed totals + measured nvidia-smi values into MEASUREMENTS.md
# § "VRAM predicted vs actual". Pass = within 20% for all three models.
