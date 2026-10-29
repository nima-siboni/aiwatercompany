# AIWaterCompany — Code Review Interview Prep

Master code review skills through structured practice. Prepare for a 45-minute spoken PR review 
with two complementary training paths: full mock PRs and focused Python pattern drills.

```
pr_mock/                    full-length mock PRs, 45 min each
  exercises/                22 mock PR files (01_data_prep.py through 22_simulation_copilot.py)
    01–04_*.py              Set A: ML and scientific code
    05–08_*.py              Set B: agent code
    09–12_*.py              Set C: ML potentials and simulation
    13–16_*.py              Set D: the discovery loop, generative models and benchmarks
    17–19_*.py              Set E: training at scale, model serving and data pipelines
    20–22_*.py              Set F: LLM pipelines (retrieval, extraction, MCP)
  answers/                  one answer key per mock PR
  learn/                    thirteen chapters with examples, plus agentic_review_checklist.md

python_patterns/            30 short drills: ten Python design patterns × three levels
  exercises/                the drills (10, 20 or 35 min per pattern, three difficulty levels)
  answers/                  answer keys, reference fixes and a one-page cheat sheet per pattern
  checks/                   runnable checks that demonstrate every planted bug
  learn/                    learning nuggets with examples, one file per pattern
```

## Quick start paths

**For interview prep in 3 hours:** Read learning nuggets for concurrency, caching, context managers, and generators (~20 min). Then tackle those four L3 pattern drills under interview conditions (35 + 5 min each).

**For thorough mastery (10 hours):** One pattern per sitting — nuggets, then L1 → L2 → L3. Finish with the 6 sets of mock PRs as dress rehearsals.

**General route:** Start with `pr_mock/learn/1_review_method.md` (applies to both). Build speed on Python patterns first (30 total), reading each pattern's nuggets first. Then do mock PR sets matching your target role. Sets C to F are specialised—pick the ones closest to your role.

## Python pattern drills (`python_patterns/`)

Build pattern recognition with 30 short drills: **10 design patterns × 3 difficulty levels**. Each level takes 10, 20, or 35 minutes. Read the nuggets first to learn the pattern, then spot the planted bugs in the drills.

### The 10 patterns covered

| Pattern | Main skills | L1 (10 min) | L2 (20 min) | L3 (35 min) |
|---------|---|---|---|---|
| **01. Decorators** | Function wrapping, composability, state | Featuriser timing | Retry backoff | Surrogate client |
| **02. Caching** | Cache keys, cache invalidation, staleness | Element lookup | Framework properties | Prediction cache |
| **03. Context managers** | Resource setup/teardown, exception safety | DFT helpers | Results store | Simulation workspace |
| **04. Generators** | Lazy evaluation, consumed-once semantics | Batched inference | Streaming dataset | GCMC ingest |
| **05. Factory & registry** | Plugin systems, late binding, registration | Featuriser factory | Filter registry | Plugin engines |
| **06. Strategy & template** | Polymorphism, composition vs inheritance | Acquisition functions | Relaxer base class | Energy fallback |
| **07. Dataclasses** | Value objects, immutability, frozen state | GCMC conditions | Frozen structures | Campaign config |
| **08. Observer & callbacks** | Event systems, loose coupling, handlers | Event bus | Training callbacks | Job pub/sub |
| **09. Singletons & DI** | Shared state, testability, hidden dependencies | Settings object | Results DB | Prediction service |
| **10. Concurrency** | Threads, locks, async/await, synchronisation | Parallel featurise | DFT queue | Async client |

Each pattern has learning nuggets (`python_patterns/learn/`), a reference fix, and runnable pytest checks that validate every planted bug.

### Which patterns for which role

- **All roles:** Start with concurrency, caching, context managers, generators (core patterns).
- **ML engineers:** Add dataclasses, observer, singleton/DI (state management).
- **Backend/infrastructure:** Add factory/registry, strategy/template (plugin systems, extensibility).
- **Full mastery:** Do all 10 patterns—they teach the same failure modes you'll see in real code.

## Mock PRs (`pr_mock/`)

**22 full-length mock PRs** (45 min each) in the style of ML-for-materials code. Each is in `pr_mock/exercises/` 
with an answer key in `pr_mock/answers/`. Each one has roughly 10–15 planted issues of mixed severity, 
plus at least one red herring (code that looks wrong but is fine), so you practise saying 
"this isn't worth blocking on".

Organized into **6 specialized sets** covering different domains and skills:
- **Set A:** ML and scientific code (data pipelines, PyTorch, numerical correctness)
- **Set B:** Agentic systems (tool loops, security, multi-agent orchestration)  
- **Set C:** ML potentials and simulation (domain-specific: forces, stresses, DFT)
- **Set D:** Discovery loops and generative models (active learning, benchmarking)
- **Set E:** Training at scale, model serving, data pipelines (DDP, FastAPI, databases)
- **Set F:** LLM pipelines (RAG, batch processing, MCP servers)

Sets C to F were built from web research into the bugs that real code in each area contains
(library docs and issue trackers, papers, post-mortems). Every planted bug was reproduced by
running code before it went into an answer key, and the keys quote those numbers.

### Learning material (`pr_mock/learn/`)

Short chapters built around small examples; every output they show came from running the
snippet. They teach the bug classes the mock PRs use, so expect to recognise some of them. The
mock PRs then train speed, ranking and explanation.

| Chapter | Read before | Time |
|---|---|---|
| [1. Reviewing out loud](pr_mock/learn/1_review_method.md): the 45 minutes, ranking, the comment formula, a worked example | everything, drills included | 10 min |
| [2. Python pitfalls](pr_mock/learn/2_python_pitfalls.md) | Set A | 10 min |
| [3. ML code](pr_mock/learn/3_ml_code.md): NumPy, pandas, PyTorch, evaluation | mock PRs 1 and 2 | 15 min |
| [4. Scientific code](pr_mock/learn/4_scientific_code.md): units, numerics, periodic structures | mock PR 3, Set C | 10 min |
| [5. Concurrency, I/O, retries, caching, security](pr_mock/learn/5_concurrency_io.md) | mock PR 4, Set E | 15 min |
| [6. Agent code](pr_mock/learn/6_agent_code.md): the loop, tools, multi-agent systems, evals | Sets B and F | 15 min |
| [Agentic review checklist](pr_mock/learn/agentic_review_checklist.md): what the research says goes wrong in agent code, and a scan order | Set B | 10 min |
| [7. ML potentials](pr_mock/learn/7_potentials.md): graphs, forces and stresses | mock PRs 9 and 10 | 15 min |
| [8. Simulation workflows](pr_mock/learn/8_simulation_workflows.md): DFT outputs, MD and the cluster | mock PRs 11 and 12 | 15 min |
| [9. The discovery loop](pr_mock/learn/9_discovery_and_evaluation.md): active learning, generative models and benchmarks | mock PRs 13, 14 and 16 | 15 min |
| [10. JAX and diffusion models](pr_mock/learn/10_jax_and_diffusion.md) | mock PR 15 | 15 min |
| [11. Training at scale](pr_mock/learn/11_training_at_scale.md): DDP, mixed precision and resumable runs | mock PR 17 | 15 min |
| [12. Services and data](pr_mock/learn/12_services_and_data.md): model APIs, databases and Parquet | mock PRs 18 and 19 | 15 min |
| [13. LLM pipelines](pr_mock/learn/13_llm_pipelines.md): retrieval, extraction and long-running assistants | Set F | 15 min |

### Set A: ML and scientific code

| # | File | Theme | Main skills tested |
|---|------|-------|--------------------|
| 1 | `pr_mock/01_data_prep.py` | Dataset loading, featurisation, splits | leakage, pandas pitfalls, mutable defaults |
| 2 | `pr_mock/02_training_loop.py` | PyTorch training + early stopping | autograd/state pitfalls, shapes, eval mode |
| 3 | `pr_mock/03_neighbour_list.py` | Crystal graph construction (PBC) | numerical/domain correctness, vectorisation |
| 4 | `pr_mock/04_simulation_runner.py` | Parallel simulation runner + cache | concurrency, error handling, I/O robustness |

Suggested order is 2 → 1 → 4 → 3, roughly easiest to hardest.

### Set B: agentic systems (based on published failure studies)

| # | File | Theme | Main skills tested |
|---|------|-------|--------------------|
| 5 | `pr_mock/05_agent_loop.py` | Tool-use loop: DB queries + cluster jobs | stop reasons, tool_result pairing, swallowed errors, loop bounds, idempotency, cache invalidation |
| 6 | `pr_mock/06_agent_tools.py` | Tool implementations for a literature agent | prompt injection, code execution, path traversal, SSRF, secret leaks, schema drift |
| 7 | `pr_mock/07_multi_agent.py` | Planner / workers / critic orchestration | termination checks, shared context, fake parallelism, verification, information withholding |
| 8 | `pr_mock/08_agent_eval.py` | Benchmark harness with an LLM judge | contamination, dropped failures, judge parsing, state leaks between episodes, noise |

Suggested order is 5 → 7 → 8 → 6. Read `pr_mock/learn/agentic_review_checklist.md` once *before*
starting Set B: it summarises what the research says goes wrong in agent code. The
exercises use the Anthropic SDK, but every bug class carries over to OpenAI or LangChain
code; answer key 05 lists the OpenAI equivalents of the API-specific checks.

### Set C: ML potentials and simulation

| # | File | Theme | Main skills tested |
|---|------|-------|--------------------|
| 9 | `pr_mock/09_potential_training.py` | Energy, force and stress training for the MOF potential | message aggregation, periodic edge vectors, autograd forces and stress, pooling, VASP stress conventions |
| 10 | `pr_mock/10_potential_finetune.py` | Fine-tuning a foundation potential, and an ASE calculator | extxyz labels, element tables, freezing, forces under `no_grad`, calculator caching |
| 11 | `pr_mock/11_dft_dataset.py` | Training set from VASP relaxations | truncated runs, constraints and force labels, atom reordering, deduplication |
| 12 | `pr_mock/12_md_diffusion.py` | CO2 diffusion from MLIP MD on Slurm | ASE units, wrapped coordinates, the MSD estimator, error bars, restarts and SIGTERM |

Suggested order is 9 → 12 → 11 → 10. Read chapters 7 and 8 first.

### Set D: the discovery loop, generative models and benchmarks

| # | File | Theme | Main skills tested |
|---|------|-------|--------------------|
| 13 | `pr_mock/13_active_learning.py` | Batch Bayesian optimisation to pick GCMC runs | GP inputs and kernels, unit conversion, pool bookkeeping, sign conventions, fair baselines |
| 14 | `pr_mock/14_generative_eval.py` | Metrics for the crystal and linker generators | periodic validity, convex-hull energies, canonical SMILES, denominators, sample sizes |
| 15 | `pr_mock/15_crystal_diffusion.py` | Crystal diffusion with classifier-free guidance (JAX) | PRNG keys, trace-time Python, guidance plumbing, sampler formulas, recompilation |
| 16 | `pr_mock/16_benchmark_report.py` | Benchmark report for a stability model | ensembles, leakage through grouping keys, coverage, bootstrap intervals, paired comparisons |

Suggested order is 13 → 16 → 14 → 15. Read chapter 9 first, and chapter 10 before 15.

### Set E: training at scale, model serving and data pipelines

| # | File | Theme | Main skills tested |
|---|------|-------|--------------------|
| 17 | `pr_mock/17_distributed_training.py` | Multi-GPU fine-tuning: DDP, AMP, EMA, resumable checkpoints | gradient sync, GradScaler, distributed evaluation, samplers, checkpoints |
| 18 | `pr_mock/18_prediction_service.py` | FastAPI service for the uptake surrogate | training/serving skew, MC dropout with BatchNorm, shared sessions, validation, per-worker state |
| 19 | `pr_mock/19_results_export.py` | Postgres upserts and a nightly Parquet export | NULLs in unique keys, upserts, watermarks, fingerprints, Parquet schemas |

Suggested order is 18 → 19 → 17. Read chapters 11 and 12 first.

### Set F: LLM pipelines

| # | File | Theme | Main skills tested |
|---|------|-------|--------------------|
| 20 | `pr_mock/20_paper_rag.py` | Retrieval over MOF synthesis papers | normalisation, filters and top-k, chunking, embedding models and prefixes, citations |
| 21 | `pr_mock/21_batch_extraction.py` | Batch extraction of adsorption data from papers | matching batch results, schemas that force values, grounding, units, truncation, caching |
| 22 | `pr_mock/22_simulation_copilot.py` | MCP server for GCMC plus a streaming chat client | units in tool code, stdio rules, tool names, cache-stable prompts, stream assembly, history trimming |

Suggested order is 21 → 20 → 22. Read chapter 13 first; chapter 6 covers the agent loop itself.

### Which PR sets for which role

AIWaterCompany is hiring for: generative models for crystals, force fields and
simulation, ML platform and infrastructure, agents with Bayesian optimisation, and internal
product engineering. If you know which role the interview is for:

- **Force fields and simulation:** Sets A and C, then 17.
- **Generative models:** Sets A and D (14 and 15 in particular), then 17.
- **ML platform or infrastructure:** Set E, then 04, 11, 12 and 19.
- **Agents and closed-loop discovery:** Sets B and F, then 13.
- **Product or backend engineering:** 18, 19, 20 and 04.

## How to run a session

1. Set a 45-minute timer and open only the exercise file, not `pr_mock/answers/`.
2. Talk out loud, or record yourself. They're grading how you think as much as what you find.
3. Spend 2 minutes on clarifying questions before you read line by line. Write them down,
   then assume reasonable answers (e.g. "this goes into the shared training pipeline").
4. Skim the whole file once, then go through it in detail. Write each comment as
   `[severity] line N: problem → concrete failure → fix`.
5. Keep the last 5 minutes for a summary: top 3 must-fix items, what tests you'd ask for,
   and what you'd let go.
6. Compare with `pr_mock/answers/NN_*.md`. Score yourself on *ranking* and *explanation*, not just the count.

## Severity tiers used in the answer keys

- **Blocker**: silently wrong results, or data loss/corruption. Must be fixed before merging.
- **Major**: crashes on realistic inputs, serious performance problems, security issues.
- **Minor**: robustness and maintainability problems that will hurt later.
- **Nit**: style, naming, small cleanups. Mention them briefly and move on.

## Talk-track template

- "Before I dive in: what is this used for, how big is the data, and is it research code or shared infra?"
- "The most important issue is X, because it gives wrong numbers *without* erroring.
  For example, with input Y you'd get Z. I'd fix it by…"
- "This one would crash, which is bad, but at least it's loud."
- "This looks like a classic bug (mutable default), but here it's never mutated, so I'd
  only flag it as a nit."
- "I'd want a test that…"
