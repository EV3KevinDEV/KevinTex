"""LFM2.5-VL backend: image → Markdown+LaTeX via Liquid AI's local VLM.

Outputs Mathpix/SimpleTex-style Markdown: prose as plain text, inline math in
`$...$`, display math in `$$...$$`. Runs entirely on-device (GPU). No cloud.
"""

import logging
import os
import re
import time

log = logging.getLogger("kevintex")

MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"

PROMPT = (
    "You are a math OCR engine. Transcribe the image to Markdown with LaTeX math.\n"
    "Rules:\n"
    "- Write normal text as plain Markdown prose.\n"
    "- Render INLINE math inside $...$ and DISPLAY math inside $$...$$.\n"
    "- Use standard LaTeX math commands only inside the math delimiters.\n"
    "- Do NOT output a LaTeX document: no \\documentclass, \\usepackage, "
    "\\begin{document}, \\end{document}, or any preamble.\n"
    "- Do NOT wrap the whole output in $$...$$.\n"
    "- Do NOT bold variables (no \\mathbf, \\boldsymbol, \\textbf). Use plain "
    "scalar notation; use \\bar{X}_n for a sample mean and Var(X) for variance.\n"
    "- Output ONLY the Markdown, no explanations, no code fences."
)

# Chain-of-thought prompt: reason first, then emit the Markdown transcription
# after a `LATEX:` sentinel. Deliberately asks for a long, multi-pass reasoning
# trace so the model spends more "thinking" tokens before committing to output.
THINKING_PROMPT = (
    "You are a meticulous math OCR engine. Transcribe the image to Markdown with LaTeX math.\n"
    "Think long and carefully before answering — take as much reasoning as you need.\n"
    "STEP 1 — OBSERVE: describe the layout of the page (title, paragraphs, displayed vs inline math, lists, alignment).\n"
    "STEP 2 — TRANSCRIBE FORMULAS ONE BY ONE: for every formula, read it left-to-right, naming each symbol, "
    "sub/superscript, fraction, root, limit, sum/integral bound, matrix cell, and alignment column. Note ambiguous "
    "glyphs (1 vs l, 0 vs O, x vs ×, v vs ν, u vs μ, ρ vs p) and resolve them from context.\n"
    "STEP 3 — VERIFY: re-read each formula against the image; fix any dropped terms, swapped indices, missing braces, "
    "or wrong delimiters. Check that every \\left is matched by a \\right and that fractions/roots have both parts.\n"
    "STEP 4 — ASSEMBLE: lay out the full document in reading order, prose as Markdown and math in delimiters.\n"
    "STEP 5 — FINAL CHECK: confirm no LaTeX preamble, no \\mathbf bolding, no code fences, and that inline math uses "
    "$...$ and display math uses $$...$$.\n"
    "STEP 6 — After all of the above reasoning, emit the final Markdown transcription on a NEW line that starts "
    "exactly with `LATEX:` followed by ONLY the Markdown.\n"
    "Rules for the final Markdown:\n"
    "- Normal text as prose; INLINE math in $...$ and DISPLAY math in $$...$$.\n"
    "- No \\documentclass, \\usepackage, \\begin{document}, preamble, or code fences.\n"
    "- Do NOT wrap the whole output in $$...$$.\n"
    "- Do NOT bold variables (no \\mathbf/\\boldsymbol/\\textbf). Use \\bar{X}_n "
    "for a sample mean and Var(X) for variance."
)

# Default generation settings from the model card.
GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.1,
    min_p=0.15,
    repetition_penalty=1.05,
)
MAX_NEW_TOKENS_FAST = 1024
MAX_NEW_TOKENS_THINKING = 3072
LATEX_SENTINEL = "LATEX:"

# Module-level thinking flag. Toggled by env var LOCALTEX_THINKING=1 at import
# time; can also be overridden per-call via LFMVisionBackend(thinking=...) or
# recognize(img, thinking=...).
THINKING = os.environ.get("LOCALTEX_THINKING", "0") in ("1", "true", "True", "yes")

_FENCES = re.compile(r"^\s*```(?:latex|tex|math|markdown)?|```\s*$", re.MULTILINE)
# Spaces the VLM tends to insert around subscript/superscript/brace tokens.
_TOKEN_SPACES = re.compile(r"\s*([_^{}])\s*")
_PREAMBLE_CMD = re.compile(
    r"\\(?:pagestyle|setlength|renewcommand|newcommand|providecommand|"
    r"title|author|date|maketitle|tableofcontents|noindent|vspace|hspace|"
    r"medskip|smallskip|bigskip|par|indent|font|usefont|selectfont)\b[^\n]*"
)


def _to_markdown_math(text: str) -> str:
    """Clean model output into Mathpix/SimpleTex-style Markdown+math."""
    # 1. Markdown fences off.
    text = _FENCES.sub("", text)
    # 2. Strip LaTeX document structure / preamble.
    text = re.sub(r"\\documentclass\s*(\[[^\]]*\])?\s*\{[^}]*\}", "", text)
    text = re.sub(r"\\usepackage\s*(\[[^\]]*\])?\s*\{[^}]*\}", "", text)
    text = re.sub(r"\\begin\s*\{document\}", "", text)
    text = re.sub(r"\\end\s*\{document\}", "", text)
    text = _PREAMBLE_CMD.sub("", text)
    text = text.strip()
    # 3. Unwrap an OUTER $$...$$ wrapper BEFORE converting inner \[...\] (so the
    #    inner conversion doesn't multiply the $$ count and block unwrapping).
    if (
        len(text) >= 4
        and text.startswith("$$")
        and text.endswith("$$")
        and text.count("$$") == 2
    ):
        inner = text[2:-2].strip()
        if inner:
            text = inner
    # 3b. Unwrap an OUTER \[...\] wrapper similarly.
    if (
        len(text) >= 4
        and text.startswith("\\[")
        and text.endswith("\\]")
        and text.count("\\[") == 1
        and text.count("\\]") == 1
    ):
        inner = text[2:-2].strip()
        if inner:
            text = inner
    # 4. Convert remaining \[...\] -> $$...$$ and \(...\) -> $...$.
    text = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", lambda m: f"$${m.group(1).strip()}$$", text)
    text = re.sub(r"\\\(\s*([\s\S]*?)\s*\\\)", lambda m: f"${m.group(1).strip()}$", text)
    # 5. De-bold variables the model over-uses (\mathbf{X}_n -> X_n).
    for cmd in ("mathbf", "boldsymbol", "textbf"):
        text = re.sub(rf"\\{cmd}\s*\{{([^{{}}]*)\}}", r"\1", text)
    # 5b. Repair orphan \left / \right tokens (model sometimes emits \right\le).
    text = re.sub(r"\\right(?!\s*[)\]|}.])", "", text)
    text = re.sub(r"\\left(?!\s*[(\[{|.])", "", text)
    # 5c. Balance: pad missing \right. for unclosed \left, drop excess \right.
    n_left = text.count(r"\left")
    n_right = text.count(r"\right")
    if n_left > n_right:
        text = text.rstrip() + " " + r" \right." * (n_left - n_right)
    # 6. Tidy token spaces inside math: \int _ { 0 } -> \int_{0}.
    prev = None
    while prev != text:
        prev = text
        text = _TOKEN_SPACES.sub(r"\1", text)
    # 7. Collapse excessive blank lines; trim ends.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _extract_markdown(text: str) -> str:
    """Pull the final Markdown out of a (possibly thinking) model response."""
    idx = text.rfind(LATEX_SENTINEL)
    if idx != -1:
        text = text[idx + len(LATEX_SENTINEL):]
    return _to_markdown_math(text)


class LFMVisionBackend:
    def __init__(self, thinking: bool | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch

        # Per-instance default; None => use module-level flag.
        self.default_thinking = THINKING if thinking is None else bool(thinking)

        t0 = time.time()
        log.info("Loading %s (first run downloads ~3.5GB weights)…", MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        self.device = next(self.model.parameters()).device
        log.info("VLM loaded in %.1fs on %s (thinking=%s)",
                 time.time() - t0, self.device, self.default_thinking)

    def _build_inputs(self, img, thinking: bool):
        prompt = THINKING_PROMPT if thinking else PROMPT
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.device)

    def __call__(self, img, thinking: bool | None = None) -> str:
        return self.recognize(img, thinking=thinking)

    def recognize(self, img, thinking: bool | None = None) -> str:
        """Recognize the formula in `img` and return cleaned LaTeX.

        If `thinking` is None, falls back to the instance default (set from
        LOCALTEX_THINKING at load time). Thinking mode uses a CoT prompt and a
        much larger `max_new_tokens` budget, then parses out the text after the
        final `LATEX:` sentinel.
        """
        think = self.default_thinking if thinking is None else bool(thinking)
        inputs = self._build_inputs(img, thinking=think)
        max_new = MAX_NEW_TOKENS_THINKING if think else MAX_NEW_TOKENS_FAST
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            **GEN_KWARGS,
        )
        gen = out[0, inputs["input_ids"].shape[-1]:]
        text = self.processor.batch_decode(
            [gen], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if think:
            return _extract_markdown(text)
        return _to_markdown_math(text)


def load():
    return LFMVisionBackend()
