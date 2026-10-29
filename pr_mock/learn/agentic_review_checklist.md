# Reviewing agentic code: what goes wrong, and what to scan for

## What the research says (digest)

- **Multi-agent failures are mostly design failures, not model failures.** MAST (Berkeley,
  NeurIPS 2025) annotated 1,642 traces from 7 multi-agent frameworks. The most common failure
  modes:
  - step repetition: 15.7%
  - reasoning–action mismatch: 13.2%
  - unaware of termination conditions: 12.4%
  - disobeying the task specification: 11.8%
  - incorrect verification: 9.1%
  - no or incomplete verification: 8.2%
- **Framework bugs are mostly ordinary bugs.** A study of 409 fixed bugs in LangChain,
  LangGraph, CrewAI, AutoGen and SmolAgents found:
  - About 52% were general programming faults: missing case handling, wrong values, type
    errors, state/lifecycle mistakes, serialisation that dropped tool-call fields, and
    async races.
  - About 22% were agent-specific: mismatches between the framework and a model's response
    format (11.7%), context/history mishandling (5.9%) and orchestration faults (3.9%).
  - 54% surfaced as crashes. The rest were quiet: wrong output, settings silently ignored,
    incomplete traces.
- **Infinite loops come from bounds that don't cover the actual cycle.** A study of 68
  confirmed infinite agent loops found 69% came from retry feedback, tool-call iteration or
  multi-agent chat with no bound. Having a `max_iterations` parameter somewhere doesn't count
  if it doesn't cover the path that actually repeats. 19 of the findings exhausted the
  context window.
- **The security risks are about content and privilege.** The OWASP Top 10 for Agentic
  Applications (Dec 2025) ranks goal hijack first: prompt injection through content the agent
  reads. Second is tool misuse: over-privileged tools, poisoned tool descriptions, or tool
  chains whose combined effect nobody authorised. Unexpected code execution and
  identity/privilege abuse are also on the list.
- **What practitioners report.** The most-cited production problem is silent tool failures
  (empty or malformed results that look like success). Tool outputs dominate token usage.
  Retrying a side effect after a timeout duplicates it, because the server may already have
  acted.

## Checklist (scan in this order)

**1. Loop control**
- Is every repeated path bounded (iterations, retries, rounds, tokens, wall-clock time)?
  Does that bound cover the cycle that actually repeats?
- Is every stop reason handled? Anthropic: `tool_use`, `end_turn`, `max_tokens`,
  `refusal`, `pause_turn`. OpenAI: `tool_calls`, `stop`, `length`, `content_filter`.
- Is termination decided in code (a boolean, a counter), not by substring-matching the
  model's prose?

**2. Message and state bookkeeping**
- Is the *full* assistant content appended (`tool_use` and thinking blocks), not just
  the text?
- Does every `tool_use` get exactly one `tool_result` with the right ID? Are they all in one
  message? With parallel execution, are results paired by ID rather than by completion order?
- Is history scoped per session, agent and episode? Watch for mutable defaults, module
  globals, and lists shared between workers.

**3. Tools**
- Are failures returned as errors (`is_error`), rather than swallowed into an empty result or
  allowed to crash the loop? Is the error message short and sanitised?
- Are inputs validated (strict schemas)? Do the schema and the function signature agree, with
  no `**kwargs` swallowing parameters?
- Do outputs have timeouts, size caps and pagination?
- For side effects: idempotency keys, no blind retries, caps, and human approval for
  irreversible or expensive actions.
- Do descriptions say when to use the tool, with units and limits, and without overlapping
  other tools?

**4. Security**
- Does one agent combine untrusted content, private data and a way to send data out (the
  "lethal trifecta")?
- Is model output treated as untrusted? That means read-only SQL, sandboxed code and shell,
  file paths confined to a workspace, SSRF checks on URLs, and an allowlist for email
  recipients.
- Are secrets kept out of prompts, tool outputs, URLs and error messages?
- Does each tool and agent have only the privileges it needs?

**5. Context and cost**
- Are tool outputs capped or summarised, and is history growth bounded (pruning or
  compaction)?
- Is the prompt cache preserved? Watch for timestamps in the system prompt, unstable tool
  ordering, and edited history.
- Are budgets enforced? Does usage accounting include cache tokens, sub-agents and judges?

**6. Concurrency**
- Sync clients inside `async` code, unbounded `gather`, rate limits, shared mutable state,
  and result ordering.

**7. Output parsing**
- Structured outputs instead of `json.loads` or regex on prose; no `content[0].text`; careful
  parsing of numbers.

**8. Multi-agent**
- Does every agent receive the task specification? (Missing it is "information
  withholding".)
- Does the verifier see the evidence, or only a summary? Prefer ground truth to another
  LLM's opinion.
- Are evaluations independent? Are generated entities deduplicated and checked against real
  data?

**9. Evaluation**
- Is the test set uncontaminated? Do failures count? Is state isolated per episode? Are runs
  repeated, with confidence intervals? Is the judge structured, calibrated and resistant to
  injection? Are per-example logs kept?

**10. Observability**
- Is every model and tool call traced (inputs, outputs, tokens, latency)? Without that,
  silent failures stay silent.

## How to prioritise out loud

1. **Silent wrong results:** mismatched tool results, swallowed errors, broken verification,
   inflated evals.
2. **Security and irreversible side effects:** injection, code execution, path traversal,
   duplicate jobs.
3. **Unbounded loops and cost.**
4. **Loud crashes:** 400s, AttributeErrors. They're bad, but at least they're visible.
5. **Hygiene:** naming, descriptions, logging.

## Sources

- [Why Do Multi-Agent LLM Systems Fail? (MAST, NeurIPS 2025)](https://arxiv.org/html/2503.13657v3)
- [Understanding Bugs in Modern Agentic Frameworks (409 bugs, 5 frameworks)](https://arxiv.org/html/2604.08906v2)
- [Bugs in Modern LLM Agent Frameworks: An Empirical Study (CrewAI, LangChain)](https://arxiv.org/html/2602.21806v2)
- [When Agents Do Not Stop: Infinite Agentic Loops](https://arxiv.org/html/2607.01641v1)
- [Verified Tool Calls Under Non-Atomic Failures](https://arxiv.org/pdf/2608.02645)
- [ReliabilityBench: agent reliability under production-like stress](https://arxiv.org/pdf/2601.06112)
- [Real Faults in MCP Software: a Taxonomy](https://arxiv.org/pdf/2603.05637)
- [OWASP Agentic Top 10 explained (Speakeasy)](https://speakeasy.com/blog/owasp-agentic-top-10-explained)
- [OWASP Agentic AI red-teaming guide (Promptfoo)](https://www.promptfoo.dev/docs/red-team/owasp-agentic-ai/)
- [AI agent failure modes: tool-calling errors, loops, propagation (Openlayer)](https://www.openlayer.com/blog/ai-agent-failure-modes-tool-calling-loops-propagation)
- [The anatomy of an agent loop (Steve Kinney)](https://stevekinney.com/writing/agent-loops)
- [The LLM agent retry bug is a distributed-systems problem](https://dev.to/tamizuddin/the-llm-agent-retry-bug-isnt-an-ai-problem-its-a-distributed-systems-problem-la5)
- [Handling stop reasons (Anthropic docs)](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
