# Answer key: 22_simulation_copilot.py

**Good clarifying questions:** Which MCP clients will use this server: only our CLI, or also
Claude Desktop and others? How long do chemists' conversations get? May the copilot start
simulations without asking (each costs cluster time)? Which force field, cutoff and charges
does the screening protocol fix? Which MOFs are in `cifs/`, and how small are their cells?

This PR has few silent bugs, and they are in the physics. Most of the rest fail loudly or cost
money, so rank them accordingly.

## Blockers

1. **The pressure is written to RASPA in bar, but RASPA reads pascals** (`RASPA_INPUT`).
   `ExternalPressure {pressure_bar}` writes 1.0 for 1 bar, and RASPA's `ExternalPressure` is in
   Pa, so every simulation runs at 1/100,000 of the requested pressure. The near-zero uptakes
   come back as the answer and the copilot reports them as uptakes at 1 bar. *Fix:* write
   `pressure_bar * 1e5`, and echo the conditions RASPA actually used next to the result.

2. **`UnitCells 2 2 2` for every MOF.** The simulation box must be at least twice the cutoff
   (24 Å with the usual 12 Å) in every perpendicular direction. Many MOF cells are much smaller
   along one axis (MOF-74's c is about 7 Å), so two cells along it are far too few: atoms
   interact with their own periodic images, and the uptakes are wrong with no error. *Fix:*
   compute the number of cells per axis from the cell's perpendicular widths and the cutoff.

## Major

3. **The stdio server writes to stdout.** Both the `print` and the RASPA child process (which
   inherits stdout) write into the protocol channel. The MCP spec says a stdio server must not
   write anything there except MCP messages. With MCP SDK 2.3, the client logged "Failed to
   parse JSONRPC message from server" for the progress line and all 20 lines of the child's
   output, then carried on, so testing looks fine. Other clients may not be so lenient, and
   output that lands in the middle of a message corrupts it. *Fix:* log to stderr, and run the
   child with `capture_output=True`, which you need anyway to report RASPA's errors.
4. **The tool names contain a dot.** `gcmc.run_gcmc` isn't a valid Anthropic tool name; tool
   names may only contain letters, digits, underscores and hyphens. The first request is
   rejected. *Fix:* sanitise the names (`gcmc__run_gcmc`) and keep a map back to (server,
   tool).
5. **The tool list is rebuilt before every model call, in completion order.**
   `asyncio.as_completed` yields the servers in whatever order they answer: over 20 turns the
   tool list came out in 2 different orders. Tools are the first part of the prompt, so each
   change of order throws away the whole prompt cache. It also adds two round trips per turn.
   *Fix:* list the tools once, refresh only when a server announces a change, and sort them by
   name.
6. **Two tool calls in one turn crash the stream parser** (`stream_turn`). `tool_json` is never
   reset between blocks, so the second tool's arguments are appended to the first's, and
   `json.loads` fails with "Extra data". Models call tools in parallel all the time. *Fix:* keep
   one buffer per block index, or better, use `client.messages.stream(...)` and append
   `get_final_message().content` unchanged. That assembles the blocks for you, and keeps any
   thinking blocks and the original block order, which the hand-built content drops.
7. **Context accounting ignores cached tokens.** With caching on, `input_tokens` counts only the
   uncached part of the prompt. In the check, a 180,050-token prompt was counted as 142 tokens,
   so `trim` never runs, and long conversations end with "prompt is too long". *Fix:* add
   `cache_creation_input_tokens` and `cache_read_input_tokens`, or call `count_tokens`.
8. **Trimming by slicing breaks the conversation** (`trim`). Once it runs, `history[-40:]`
   drops the user's original request. Whenever the cut lands on a tool-result turn, the history
   starts with an orphaned tool result (for tool call `t2` in the check), which the API
   rejects. And trimming a few messages at a time changes the cached prefix on every turn.
   *Fix:* cut only before a plain user turn, keep the first request (or a summary), and trim in
   large steps.
9. **Tool output is unbounded, and errors aren't flagged.** Every call puts the whole RASPA
   output file into the conversation: force-field tables, cycle reports and averages, when the
   model needs one uptake. When RASPA fails, the MCP server returns `is_error=True` with only
   "Error executing tool run_gcmc", because SDK 2.x hides the exception's text, and the client
   drops the flag. *Fix:* return the uptake, its error bar, units and conditions as a small
   structured result; raise the MCP `ToolError` with RASPA's message; and pass `is_error`
   through to the `tool_result`.

## Minor

10. **Output tokens are summed over `message_delta` events,** which are cumulative: 92 counted,
    against 52 really. Take the last value.
11. **The tool blocks for about two minutes.** Anything longer needs submit-and-poll tools, or
    the call will hit client timeouts.
12. **`cifs/{mof}.cif` is built from model input,** so a prompt-injected `../` path can read
    other files. Validate the name against the known MOFs.
13. **`input()` blocks the event loop.** That's acceptable for a CLI, but nothing from the
    servers is processed while it waits.

## Red herrings (say why they're fine)

- **`AsyncExitStack` for several stdio clients** is the right way to keep them all open and
  close them cleanly.
- **Top-level `cache_control={"type": "ephemeral"}`** (automatic caching) is right for a
  growing conversation, once the prefix stops changing (5, 8).
- **Raising inside the tool** is fine: the SDK turns the exception into an error result. The
  problems are the hidden message and the dropped flag (9).
- **`json.loads(tool_json or "{}")`** handles tools called with no arguments.
- **Passing `block["input"]` straight to `call_tool`** is fine: it's a dict.

## Model 1-minute summary
"The two silent bugs are in the physics. RASPA's `ExternalPressure` is in pascals and the tool
writes bar, so every simulation runs at a hundred-thousandth of the requested pressure. And
`UnitCells 2 2 2` is too small a box for MOFs with short axes, so their uptakes are wrong too.
On the client: dotted tool names are rejected by the API; the tool list is rebuilt in random
order every turn, which defeats the prompt cache; two tool calls in one turn crash the stream
parser; context accounting ignores cached tokens, so trimming never runs, and when it does, it
orphans tool results and drops the user's request. The server also writes to stdout, which the
spec forbids, though this client tolerates it. I'd ask for an end-to-end test with a fake RASPA
that checks the pressure in the input file, plus a recorded two-tool-call stream."
