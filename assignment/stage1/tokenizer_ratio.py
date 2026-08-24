"""
Stage 1A — Tokenizer forensics: turn "Vietnamese costs more tokens" into a
number you own.

Measures, per model, how many more tokens Vietnamese legal text costs than its
English translation, then converts that into an EFFECTIVE context window
(advertised window / ratio) — the number that actually matters when you budget
tokens for a Vietnamese-language system.

Run:
    python tokenizer_ratio.py

Requires: transformers. Three of the four models are GATED on Hugging Face —
you must accept each model's license on its Hub page and be logged in
(`huggingface-cli login`). Any model that fails to load is skipped with a note,
so the script still runs on whatever you have access to.
"""

from transformers import AutoTokenizer

# 10 Vietnamese/English pairs from the maritime-law domain. Ten+ is required:
# legal text tokenizes very differently from conversational text, and a single
# sentence gives a noisy ratio (the assignment's trap).
PAIRS = [
    ("Thuyền trưởng phải chịu trách nhiệm về an toàn của tàu biển.",
     "The captain shall be responsible for the safety of the seagoing vessel."),
    ("Chủ tàu có nghĩa vụ bảo đảm tàu biển đủ khả năng đi biển trước mỗi chuyến đi.",
     "The shipowner is obliged to ensure the vessel is seaworthy before each voyage."),
    ("Thuyền viên phải tuân theo mệnh lệnh của thuyền trưởng trong phạm vi chức trách.",
     "Crew members must obey the master's orders within the scope of their duties."),
    ("Hợp đồng vận chuyển hàng hóa bằng đường biển phải được lập thành văn bản.",
     "A contract for the carriage of goods by sea must be made in writing."),
    ("Người vận chuyển chịu trách nhiệm về tổn thất hàng hóa xảy ra trong thời gian vận chuyển.",
     "The carrier is liable for loss of goods occurring during the period of carriage."),
    ("Cảng vụ hàng hải có quyền không cho tàu rời cảng nếu phát hiện vi phạm.",
     "The maritime port authority may prevent a ship from leaving port if a violation is found."),
    ("Tàu biển phải được đăng ký trong Sổ đăng ký tàu biển quốc gia.",
     "A seagoing vessel must be registered in the national register of ships."),
    ("Việc cứu hộ hàng hải được thực hiện trên cơ sở hợp đồng cứu hộ.",
     "Maritime salvage shall be carried out on the basis of a salvage contract."),
    ("Tổn thất chung là những hy sinh và chi phí bất thường được thực hiện một cách cố ý và hợp lý.",
     "General average is the extraordinary sacrifice and expenditure made intentionally and reasonably."),
    ("Thời hiệu khởi kiện về hư hỏng hàng hóa là một năm kể từ ngày trả hàng.",
     "The statute of limitations for claims of cargo damage is one year from the date of delivery."),
]

MODELS = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
]

# Advertised (native) context windows, for computing the effective window.
# Note some models advertise a larger window only with RoPE scaling (e.g. YaRN);
# these are the commonly cited native figures — adjust to what you'll actually run.
ADVERTISED_CTX = {
    "Qwen/Qwen3-8B": 32_768,
    "meta-llama/Llama-3.1-8B-Instruct": 131_072,
    "google/gemma-3-12b-it": 131_072,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 131_072,
}


def n_tokens(tok, text):
    """Token count WITHOUT special tokens, so ratios compare raw content."""
    return len(tok(text, add_special_tokens=False).input_ids)


def main():
    print(f"{'Model':<50} {'ratio':>6} {'vocab':>8} {'adv ctx':>9} {'eff ctx':>9}")
    print("-" * 86)

    first_ok_tok = None
    for m in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(m)
        except Exception as e:  # gated / not logged in / offline
            print(f"{m:<50}   SKIPPED — {type(e).__name__}: {str(e).splitlines()[0][:60]}")
            continue

        ratios = [n_tokens(tok, vi) / n_tokens(tok, en) for vi, en in PAIRS]
        mean_ratio = sum(ratios) / len(ratios)
        adv = ADVERTISED_CTX.get(m, 0)
        eff = int(adv / mean_ratio) if adv else 0
        print(f"{m:<50} {mean_ratio:>6.2f} {tok.vocab_size:>8} {adv:>9,} {eff:>9,}")

        if first_ok_tok is None:
            first_ok_tok = (m, tok)

    # ── Diacritics: decode one Vietnamese sentence token by token ─────────────
    # Look at where the diacritics land — Vietnamese diacritics (dấu) often get
    # split onto their own byte-level tokens or fragment a syllable, which is
    # exactly why the VI/EN ratio is > 1.
    if first_ok_tok is not None:
        name, tok = first_ok_tok
        vi = PAIRS[0][0]
        ids = tok(vi, add_special_tokens=False).input_ids
        print(f"\nToken-by-token decode of one VI sentence with {name}:")
        print(f"  {vi}")
        print(f"  {len(ids)} tokens\n")
        # raw BPE tokens: `Ġ` = a leading space; the mojibake is byte-level
        # UTF-8 (each byte of a multi-byte char shown as a placeholder glyph),
        # NOT corruption. This is where you SEE the diacritics fragment.
        print("  raw tokens:  " + " | ".join(tok.convert_ids_to_tokens(ids)))
        # same ids decoded back to readable Vietnamese, so the split points are
        # legible: watch syllables like "Thuyền" break into "Th" + "uyền".
        readable = [repr(tok.decode([i])) for i in ids]
        print("  readable:    " + " | ".join(readable))


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────────────────────────────────────
#
# Effective context window = advertised window / mean VI/EN ratio. If a model
# advertises 32K tokens but Vietnamese legal text costs 1.8x English, your real
# working budget for that content is ~18K tokens. That single number decides
# whether a document fits — advertised windows lie for non-English text.
#
# Trap avoided: the ratio is averaged over 10 domain sentences, not one. Legal
# Vietnamese (dense noun phrases, Sino-Vietnamese compounds) tokenizes worse
# than conversational Vietnamese, so a chat-derived ratio would understate cost.
#
# Fill the printed numbers into MEASUREMENTS.md § "Tokenizer forensics".
