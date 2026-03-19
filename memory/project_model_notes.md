---
name: EEG-FM-Bench registered model notes
description: Which models auto-download weights vs need local paths, and their sampling rates
type: project
---

Models with HuggingFace auto-download (no local weights needed):
- ZUNA: Zyphra/ZUNA — 256 Hz, frozen by default
- MOMENT: google/flan-t5-small backbone — 256 Hz
- MANTIS: HF download — 256 Hz

Models requiring local pretrained weights (no auto-download):
- BENDR: pretrained_path (contextualiser .pt) + pretrained_conv_path (conv encoder .pt) — 256 Hz
- BIOT: pretrained_path (.pt) — 200 Hz (needs separate 200 Hz preprocessing)
- LABRAM: pretrained_path — 200 Hz
- CBraMod: pretrained_path — 200 Hz
- REVE: pretrained_path + pos_bank_pretrained_path — 200 Hz
- EEGPT: pretrained_path — 256 Hz
- CSBrain: pretrained_path — 200 Hz

Without pretrained weights, freeze_encoder=true gives frozen random encoder (valid control baseline, not paper-equivalent).

Commented-out in registry (unavailable): EEGNet, EEGConformer.

**Why:** Critical when reproducing Table 4 — only ZUNA works out-of-box; others need weights for paper-equivalent results.
**How to apply:** When adding more models to benchmark, check this list for weight requirements and sampling rate (affects preprocessing).
