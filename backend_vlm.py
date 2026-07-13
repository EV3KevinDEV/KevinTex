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
    "OCR task: transcribe ONLY visible text and formulas from the image.\n"
    "Reply with exactly ONE line in this format:\n"
    "LATEX: <markdown transcription>\n"
    "Rules for the markdown after `LATEX:`:\n"
    "- Plain text for prose; inline math in $...$; display math in $$...$$.\n"
    "- Copy only what is visible. No guessing or filling in missing parts.\n"
    "- No explanations, analysis, notes, headings, apologies, or task descriptions.\n"
    "- No \\documentclass, \\usepackage, preamble, code fences, or outer $$...$$ wrapper.\n"
    "- No \\mathbf/\\boldsymbol/\\textbf bolding.\n"
    "- If blank or unreadable, reply exactly: LATEX:"
)

# Thinking mode keeps reasoning off the final answer via a `LATEX:` sentinel.
THINKING_PROMPT = (
    "You are a strict OCR engine. Transcribe ONLY the text and formulas visible in the image.\n"
    "Reason silently about symbols and layout, but NEVER print that reasoning.\n"
    "Your entire visible reply must be exactly one line starting with `LATEX:` "
    "followed by only the final Markdown transcription.\n"
    "Rules for the Markdown after `LATEX:`:\n"
    "- Transcribe only visible content from the image.\n"
    "- No commentary, analysis, notes, headings, apologies, or task descriptions.\n"
    "- INLINE math in $...$ and DISPLAY math in $$...$$.\n"
    "- No \\documentclass, \\usepackage, preamble, code fences, or outer $$...$$ wrapper.\n"
    "- No \\mathbf/\\boldsymbol/\\textbf bolding.\n"
    "- If unreadable, output exactly `LATEX:` with nothing after it."
)

AUDIO_PROMPT = (
    "Listen to the spoken mathematical expression and convert it to Markdown + LaTeX.\n"
    "Reply with exactly ONE line in this format:\n"
    "LATEX: <markdown transcription>\n"
    "Interpret spoken math structurally: for example, 'x squared' is $x^2$, "
    "'one half' is $\\frac{1}{2}$, and spoken limits, sums, roots, matrices, "
    "Greek letters, subscripts, and superscripts should use standard LaTeX.\n"
    "Use plain text only for clearly dictated prose, inline math in $...$, and "
    "display math in $$...$$. Do not explain, solve, or add content.\n"
    "If the audio is blank or unintelligible, reply exactly: LATEX:"
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
_META_LINE = re.compile(
    r"^\s*(?:\#{1,6}\s*|\*\*)?"
    r"(?:analysis(?:\s+of(?:\s+the)?\s+content)?|note|observation|"
    r"self[- ]?correction(?:/refinement)?|refinement|"
    r"text\s+transcription|transcription(?:\s+based)?|warning|important|"
    r"step\s*\d+|here is|below is|the (?:text|input|user|image)|"
    r"this (?:seems|appears|is)|if (?:this were|the image|unreadable))"
    r"(?:\s+of\s+the\s+content)?(?:\*\*)?\s*:?",
    re.I | re.M,
)
_META_PHRASES = (
    "highly fragmented",
    "lacks context",
    "coherent document",
    "collection of notes",
    "literal transcription",
    "self-correction",
    "structured document",
    "visible text",
    "user input",
    "study notes",
    "fragmented and lacks",
    "transcription above",
    "transcription below",
    "the input is",
    "the input seems",
    "the text provided",
    "snippet from",
    "likely related",
    "possibly related",
    "mathematical derivation",
    "does not provide enough context",
    "difficult to reconstruct",
    "not enough context",
    "based on the visible",
)
_PREAMBLE_LINE = re.compile(
    r"^\s*(?:the\s+)?(?:text|image|content)\s+(?:provided|appears|contains|does not|is)\b",
    re.I,
)
_COMMENTARY_LINE = re.compile(
    r"\b(?:snippet|context|incomplete|unreadable|commentary|describe|summarize|"
    r"explain|suggests a reference|strongly suggests)\b",
    re.I,
)


def _has_math_markers(text: str) -> bool:
    return bool(re.search(r"\$|\\\(|\\\[|\\begin\{", text))


def _looks_like_meta_junk(text: str) -> bool:
    lowered = text.lower()
    hits = sum(1 for phrase in _META_PHRASES if phrase in lowered)
    if _META_LINE.search(text):
        return True
    if _PREAMBLE_LINE.search(text):
        return True
    if re.search(r"\bhere is the transcription\b", lowered):
        return True
    if hits >= 1 and not _has_math_markers(text) and len(text) > 80:
        return True
    if hits >= 2 and not _has_math_markers(text):
        return True
    if hits >= 3:
        return True
    return False


def _is_meta_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _META_LINE.match(stripped):
        return True
    if _PREAMBLE_LINE.match(stripped):
        return True
    lowered = stripped.lower()
    if any(phrase in lowered for phrase in _META_PHRASES):
        return True
    if _COMMENTARY_LINE.search(stripped) and not _has_math_markers(stripped):
        return True
    if re.match(r"^\*\*[^*]+\*\*\s*:?\s*$", stripped):
        return True
    if re.match(r"^\(.*\)$", stripped) and len(stripped) > 30:
        return True
    return False


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        key = re.sub(r"[*_`\"']", "", line).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return kept


def _has_substantive_ocr(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _is_meta_line(stripped):
            continue
        if re.sub(r"[\s$`*_#>.-]", "", stripped):
            return True
    return False


def _filter_ocr_lines(text: str) -> str:
    """Keep only lines that look like transcribed content, not model commentary."""
    quoted = re.findall(r'"([^"]{2,})"|“([^”]{2,})”', text, flags=re.S)
    if quoted:
        parts: list[str] = []
        for q in quoted:
            block = (q[0] or q[1]).strip()
            for line in block.splitlines():
                stripped = line.strip()
                if stripped and not _is_meta_line(stripped):
                    parts.append(stripped)
        if parts:
            return "\n".join(_dedupe_lines(parts))

    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _is_meta_line(stripped):
            continue
        stripped = re.sub(r'^["\']|["\']$', "", stripped).strip()
        stripped = re.sub(r"^\*([^*]+)\*$", r"\1", stripped).strip()
        if stripped and not _is_meta_line(stripped):
            kept.append(stripped)
    return "\n".join(_dedupe_lines(kept)).strip()


def _strip_meta_junk(text: str) -> str:
    """Drop model commentary that leaked into the OCR answer."""
    if not text.strip():
        return ""

    if (
        len(text) >= 4
        and text.strip().startswith("$$")
        and text.strip().endswith("$$")
        and text.count("$$") == 2
    ):
        text = text.strip()[2:-2].strip()

    cleaned = _filter_ocr_lines(text)
    if _has_substantive_ocr(cleaned):
        return cleaned
    return ""


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
    return _strip_meta_junk(text)


def _extract_markdown(text: str) -> str:
    """Pull the final Markdown out of a (possibly thinking) model response."""
    idx = text.rfind(LATEX_SENTINEL)
    if idx != -1:
        text = text[idx + len(LATEX_SENTINEL):].lstrip()
    elif _looks_like_meta_junk(text):
        text = _filter_ocr_lines(text)
        if not _has_substantive_ocr(text):
            return ""
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
        return _extract_markdown(text)


def load():
    return LFMVisionBackend()
