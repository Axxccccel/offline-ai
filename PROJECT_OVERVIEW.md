# Offline AI Assistant (v7) — Project Overview

> Handoff document for another AI/developer. Describes what this project is,
> how it is built, how the request pipeline works, what has been done, and
> what remains. Written 2026-08-26.

## 1. What this is

A **fully local, offline, private AI assistant** for Windows with a Tkinter
GUI. It runs a 9B language model on the user's own GPU (NVIDIA RTX 2060, 6 GB
VRAM) and answers everyday questions — knowledge, explanations, reasoning,
advice, coding, conversation, images — **without sending any data to the
cloud**. For questions that need *current* facts (today's leaders, prices,
weather, sports, events) it uses a local retrieval pipeline (Wikipedia /
Wikidata / Brave web search / live APIs) to ground its answers. Everything
except live web fetches happens on-device.

The core value proposition: **cloud-model-level quality on everyday tasks,
but 100 % private and offline, with no datacenter and no per-query cost.**

## 2. Status at a glance

- **Chat & code model:** Qwen3.5-9B (Q4_K_M, GGUF) — one model for everything,
  including code (a former dedicated Coder-7B was removed; see §8).
- **Vision model:** Qwen2.5-VL-3B, run via a local **Ollama** server for image
  questions.
- **Retrieval reranker:** bge-small-en-v1.5 (tiny embedding model).
- **Benchmark (300 fresh questions, 14 languages, 12 conversations):**
  283/300 = **94 %** strict (~96 % adjusted for grading artifacts).
  Non-live core capability 95 %; live/current facts 86 %; conversations 98 %.
- **A/B vs a bare Qwen 9B (same model, no pipeline):** the pipeline adds
  **+30 pp on live/current facts** and is roughly neutral elsewhere (the base
  model is already strong on knowledge/reasoning/advice/code/conversation).

## 3. Hardware & the single-model constraint

- GPU: NVIDIA RTX 2060, **6 GB VRAM**. Acceleration via Vulkan (llama.cpp).
- **CRITICAL:** only **one** LLM instance fits in 6 GB. Loading a second (e.g.
  running a headless test while the GUI exe is also running) causes 5–40×
  slowdown from VRAM/GPU contention. **Always stop the running exe before any
  headless/model run**, and relaunch afterwards.
- Main model load config ("snabb"): `n_ctx=4096, flash_attn=True, type_k=8,
  type_v=8, n_gpu_layers=-1` (all layers on GPU, with a fallback ladder down
  to CPU).

## 4. File map (repo root)

| File / dir | Role |
|---|---|
| `AIPrototype.py` | **The whole app** (~7,300 lines): GUI, pipeline, routing, retrieval glue, guards. Start here. |
| `retrieval_deep.py` | Deep retrieval: fetch-and-read pages + bge-small passage reranker (`djup_kontext`). |
| `sprak.py` | Language helpers / tables. |
| `rth_stdio.py` | Small stdio helper. |
| `AIPrototype_v7.exe` | Built app (PyInstaller onefile). **Run this one — it must sit next to `models\`.** |
| `AIPrototype_v7.spec` | PyInstaller build spec. |
| `_rebuild.ps1` | Build script: py_compile → kill exe → PyInstaller → copy `dist\` exe to root → relaunch. |
| `models\` | GGUF model files (see §5). |
| `ikon.ico` | App icon. |
| `build\`, `dist\` | PyInstaller intermediates (regenerated on build; only needed to rebuild). |
| `AIPrototype_pre_*.bak` | Timestamped source backups (one per major change). |
| `_ta_bort\` | "To delete" holding folder: old scratch, test/eval scripts, logs, old backups, removed models. Safe to delete after review. |

The reusable **evaluation harness** lives in `_ta_bort\` (moved during a
cleanup): `_eval300.py`, `_evalset300.json`, `_grade300.py`, plus the 200/50
variants and the A/B `_eval300_vanilla.py` + `_grade_compare.py`. Pull them
back out if you want to re-benchmark.

## 5. Models (`models\`)

| File | ~Size | Purpose |
|---|---|---|
| `Qwen3.5-9B-Q4_K_M.gguf` | 5.3 GB | Main model — chat **and** code. |
| `Qwen2.5-VL-3B-Instruct-q4_k_m.gguf` | 1.8 GB | Vision model (images), **via Ollama**. |
| `Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf` | 1.25 GB | The VL model's multimodal projector (its "eyes"). Required companion — not a separate model. |
| `bge-small-en-v1.5-f16.gguf` | 65 MB | Retrieval passage reranker. Small but **functionally required** by `retrieval_deep.py`. |

Vision runs through a local Ollama server (`qwen2.5vl:3b`), started on demand
by `_sakerstall_ollama()`; see `beskriv_bild()`. Text and images are separate
paths; the text pipeline never calls the vision model.

## 6. Build & run

**Run:** launch `AIPrototype_v7.exe` **from the folder that contains
`models\`** (i.e. the repo root, or `C:\AIBuild\AIPrototype_v7.exe`). The app
computes its base dir as `os.path.dirname(sys.executable)` when frozen, so the
exe must sit directly beside `models\`. Do **not** run the copy inside `dist\`.

**Rebuild:** run `_rebuild.ps1` (PowerShell). It compiles, kills the running
exe, builds the onefile with `AIPrototype_v7.spec`, copies `dist\` exe to root,
and relaunches. Build takes a few minutes and the onefile is ~236 MB (it
bundles Python + llama-cpp + deps).

## 7. The request pipeline (`_fraga_ai_inner` in `AIPrototype.py`)

Every user turn flows through an ordered set of routers/guards. Earlier
branches are cheaper/more deterministic; retrieval and free generation come
last. Approximate order:

1. **Code router** — `_ar_kodfraga` (+ hysteresis flag `_SENAST_KOD`) → answer
   with the 9B model using the code system prompt `_KOD_SYSTEM` (`_svara_kod`).
2. **Follow-up rewrite** — resolve anaphora against history (`_ar_foljdfraga` /
   `_skriv_om_foljdfraga`).
3. **Direct translation** requests → dedicated path.
4. **`svarssprak`** = `detektera_svarssprak_namn(...)` — the answer's target
   language (the one place language is decided).
5. **Exact calculator** — pure arithmetic computed in Python (`_berakna`).
6. **Calendar helper (A5)** — weekday math / period constants (`_svara_datumfraga`).
7. **Synthesis turn (Grepp 4)** — "summarize / make a pitch / a plan" in an
   ongoing conversation → answer from history only, never search.
8. **Quantitative follow-up (Grepp 3)** — compute from earlier numbers locally.
9. **Personal / emotional** message → warm local answer, no search.
10. **Office-holder lookup** — deterministic Wikidata P6/P35 (today's leader).
11. **Native simple multilingual fact (Do-no-harm Fix 1)** — short non-en/sv
    factual questions answered natively (avoids exonyms from the English
    retrieval detour).
12. **Structured Wikidata** — creators (P170/P50), capital (P36), dates, etc.
13. **Champion/winner** extraction.
14. **Intent routing (Grepp 5)** — concept ("explain X") and advice
    ("should I…") answered locally, not deflected/over-hedged.
15. **Retrieval** — if `behover_aktuell_info` says the question needs current
    data: office/Wikipedia/Wikidata/Brave, then a deep fetch-and-read path.
16. **Relevance gate (A1)** — drop unrelated blocks from raw search context.
17. Speculative/opinion, yes-no, list, and fact extraction sub-paths.
18. **Empty-retrieval memory fallback (Do-no-harm Fix 2)** — if a current
    question got zero grounding, answer from the model's memory **with a
    "may be outdated, verify" caveat** instead of a flat "I don't know"
    (excludes future/release-date questions).
19. **Grounded generation with guards** — two attempts (`svar1`/`svar2`) filtered
    by broken-answer / hedge / grounding / hallucination checks, then honest
    fallbacks. Answer language mirrored via `_sprak_direktiv`.

## 8. What has been built (change history)

Each "grepp" (move) targets a measured failure pattern; all verified with zero
regressions and documented in the attached Claude **project** (docs named
`v7_*.md`).

- **Grepp 1** — answer-contract + language gate (definition-dumps → answer
  first; stop language drift).
- **Grepp 2** — structured grounding (correct Wikidata property per verb) +
  a standing 50-question benchmark harness.
- **Grepp 3** — three cluster fixes: broader role lookup (VP, EU bodies,
  Spain, Poland), near-language pair gate (pt/es, sv/da), local arithmetic in
  conversation.
- **Grepp 4** — synthesis turn (final "summarize/pitch/plan" answered from the
  conversation, never search; killed a cross-topic hallucination).
- **Grepp 5** — intent routing (concept + advice answered locally; ends
  concept-refusal and advice over-hedging).
- **A1–A6** (quality/robustness): A1 relevance gate on retrieved context;
  A2 adaptive output-token cap (up to 1400, shrinks to fit context) +
  sentence-clean truncation; A3 rolling conversation summary for long threads;
  A4 season-aware sports retrieval; A5 deterministic calendar helper;
  A6 force grounding for specific-event questions.
- **"Do no harm"** (from the A/B vs bare model): Fix 1 native simple
  multilingual facts; Fix 2 memory-with-caveat on empty retrieval.
- **Code → 9B** — the dedicated Qwen2.5-Coder-7B and the GPU model-swap were
  removed; code is now answered by the 9B with `_KOD_SYSTEM`. Faster (2–4 s vs
  ~15–25 s swap) and simpler.

## 9. Known limitations / roadmap

Real, measured (small) failure patterns still open:

- **Reasoning slips** — occasional single-step arithmetic errors mid-chain.
  *Planned:* a lightweight verify/recompute pass on reasoning answers.
- **Creative language drift** — Swedish poems can drift to Norwegian (near
  language). The language gate covers factual answers but not free/creative
  text. *Planned:* extend the gate to creative output.
- **Event over-specification** — thin event data can still yield a fabricated
  specific event. *Planned:* harder honesty gate when event retrieval is thin.
- **Transient empty retrieval** — occasional empty web result on a major
  entity (looks transient). Fix 2 mitigates; a retry-on-empty would help.

Product track (not started): licensing/activation, signed installer with
auto-update, onboarding, an automated benchmark run per build.

## 10. Gotchas (read before editing/running)

- **One model on the GPU** — kill `AIPrototype_v7.exe` before any headless run;
  relaunch after. (See §3.)
- **Exe must sit beside `models\`** — run the root exe, not `dist\`.
- **PowerShell inline `$` variables get mangled** through the shell layer here
  — write `.ps1` script files instead of inline `$var` commands.
- **Retrieval needs network**; everything else is offline. Transient web
  throttling can cause one-off empty results — not necessarily a code bug.
- **Headless test pattern:** `exec` the module top with
  `src.split("root = tk.Tk()")[0]`, load the model once, set
  `ns["llm"] = ns["llm_sok"] = wrapper`, then call `ns["fraga_ai"]`.
- The app writes `debug.log` in its base dir (enable `_debug_logg`).

## 11. Where the detailed history lives

Full per-change write-ups (rationale, code, verification, benchmark numbers)
are in the attached Claude **project** as ~20 Swedish markdown docs named
`v7_*.md` and `v8_*.md` (e.g. `v7_300q_revision.md`,
`v7_app_vs_vanilla_jamforelse.md`, `v7_gor_ingen_skada.md`,
`v7_kod9b_och_stadning.md`). This file is the English summary of that history.
