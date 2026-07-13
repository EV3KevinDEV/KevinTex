"""Tests for OCR output cleanup and instruction-following guards."""

import unittest

from backend_vlm import _extract_markdown, _to_markdown_math


class BackendCleanupTests(unittest.TestCase):
    def test_meta_commentary_is_removed(self):
        junk = """$$
The text provided is highly fragmented and appears to be a collection of notes.

**Analysis:** The input seems to be a collection of fragmented notes.

**Note:** The transcription above is a literal transcription of the visible text.

**Self-Correction/Refinement:** The input seems to be a collection of notes.
$$"""
        self.assertEqual(_to_markdown_math(junk), "")

    def test_user_physics_snippet_is_stripped(self):
        junk = """The text provided is a snippet from a mathematical or physics problem, likely related to a physics or mathematics context. The content appears to be a set of equations or definitions, possibly related to a physics problem or a mathematical derivation.

Here is the transcription based on the visible text:

**Text Transcription:**

"Larsen and Marx 5.2.1.
Theorems 1.
... (This section seems incomplete or highly fragmented.)"

**Analysis of the content:**

The image contains fragments of mathematical notation, likely related to physics or mathematics. The text is very fragmented and lacks context, making it difficult to reconstruct a complete equation.

The visible text suggests a reference to specific mathematical concepts or theorems, possibly related to physics or mechanics. The structure strongly suggests a mathematical problem or a derivation.

**Transcription based on the visible text:**

*Larsen and Marx 5.2.1.*
*Theorems 1.*

(This is a highly fragmented transcription based on the visible text.)

If the image contains complex equations, the transcription provided above is incomplete. The image does not provide enough context to reconstruct a full mathematical expression."""
        self.assertEqual(
            _extract_markdown(junk),
            "Larsen and Marx 5.2.1.\nTheorems 1.",
        )

    def test_real_math_survives_cleanup(self):
        raw = "Let $x^2 + 1 = 0$ and $$\\int_0^1 f(x)\\,dx.$$"
        self.assertEqual(_to_markdown_math(raw), raw)

    def test_thinking_without_sentinel_drops_meta_junk(self):
        junk = (
            "Analysis: fragmented notes without formulas.\n"
            "Note: lacks context and is not a coherent document."
        )
        self.assertEqual(_extract_markdown(junk), "")

    def test_thinking_with_sentinel_keeps_only_transcription(self):
        raw = (
            "Reasoning hidden.\n"
            "LATEX: The limit is $$\\lim_{n\\to\\infty} \\frac{1}{n} = 0.$$"
        )
        self.assertIn("\\lim", _extract_markdown(raw))


if __name__ == "__main__":
    unittest.main()
