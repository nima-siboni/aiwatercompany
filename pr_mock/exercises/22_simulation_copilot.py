"""
PR #301: Simulation copilot: a GCMC MCP server and a streaming chat client

Two parts.
1. An MCP server (stdio) that lets the copilot run short RASPA GCMC simulations
   (`python copilot.py serve`). Each run takes about two minutes.
2. The chat client the chemists use. It starts every MCP server listed in SERVERS, offers all of
   their tools to Claude as "<server>.<tool>", streams the answer to the terminal, and trims old
   turns when the conversation gets close to the context limit.
"""
import asyncio
import glob
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import AsyncExitStack

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer

# ------------------------------------------------------------------------- 1. MCP server
mcp = MCPServer("gcmc")

RASPA_INPUT = """SimulationType MonteCarlo
NumberOfCycles {cycles}
NumberOfInitializationCycles {init}
Framework 0
FrameworkName {name}
UseChargesFromCIFFile yes
UnitCells 2 2 2
ExternalTemperature {temperature_k}
ExternalPressure {pressure_bar}
Component 0 MoleculeName CO2
            MoleculeDefinition TraPPE
            TranslationProbability 0.5
            ReinsertionProbability 0.5
            SwapProbability 1.0
            CreateNumberOfMolecules 0
"""


@mcp.tool()
def run_gcmc(mof: str, pressure_bar: float, temperature_k: float = 298.0) -> str:
    """Run a short CO2 GCMC simulation in one MOF and return the RASPA output."""
    work = tempfile.mkdtemp(prefix="gcmc_")
    shutil.copy(f"cifs/{mof}.cif", work)
    with open(f"{work}/simulation.input", "w") as f:
        f.write(RASPA_INPUT.format(cycles=2000, init=1000, name=mof,
                                   temperature_k=temperature_k, pressure_bar=pressure_bar))
    print(f"running RASPA: {mof} at {pressure_bar} bar, {temperature_k} K")
    subprocess.run(["simulate", "simulation.input"], cwd=work, check=True)
    with open(glob.glob(f"{work}/Output/System_0/*.data")[0]) as f:
        return f.read()


# ------------------------------------------------------------------------- 2. chat client
SERVERS = {"gcmc": [sys.executable, "copilot.py", "serve"],
           "matdb": [sys.executable, "materials_db_server.py"]}
MODEL = "claude-opus-5-5"
MAX_CONTEXT = 150_000
SYSTEM = "You help chemists screen MOFs for CO2 capture. Use the tools; report numbers with units."
client = anthropic.AsyncAnthropic()
used_tokens = 0


async def list_tools(name, session):
    return name, await session.list_tools()


async def load_tools(sessions):
    tools, routes = [], {}
    for fut in asyncio.as_completed([list_tools(n, s) for n, s in sessions.items()]):
        name, listing = await fut
        for t in listing.tools:
            qualified = f"{name}.{t.name}"
            tools.append({"name": qualified, "description": t.description or "", "input_schema": t.input_schema})
            routes[qualified] = (sessions[name], t.name)
    return tools, routes


async def stream_turn(tools, history):
    global used_tokens
    content, text, tool_json, current, stop_reason = [], "", "", None, None
    usage = {"input": 0, "output": 0}
    stream = await client.messages.create(model=MODEL, max_tokens=8000, system=SYSTEM, tools=tools,
                                          messages=history, cache_control={"type": "ephemeral"},
                                          stream=True)
    async for event in stream:
        if event.type == "message_start":
            usage["input"] = event.message.usage.input_tokens
        elif event.type == "content_block_start" and event.content_block.type == "tool_use":
            current = {"type": "tool_use", "id": event.content_block.id, "name": event.content_block.name}
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                text += event.delta.text
                print(event.delta.text, end="", flush=True)
            elif event.delta.type == "input_json_delta":
                tool_json += event.delta.partial_json
        elif event.type == "content_block_stop" and current is not None:
            current["input"] = json.loads(tool_json or "{}")
            content.append(current)
            current = None
        elif event.type == "message_delta":
            stop_reason = event.delta.stop_reason
            usage["output"] += event.usage.output_tokens
    if text:
        content.insert(0, {"type": "text", "text": text})
    used_tokens = usage["input"] + usage["output"]
    return content, stop_reason


def trim(history, keep=40):
    return history[-keep:] if used_tokens > MAX_CONTEXT else history


async def chat_turn(sessions, history, user_text):
    history.append({"role": "user", "content": user_text})
    while True:
        tools, routes = await load_tools(sessions)  # servers can add tools while we run
        history[:] = trim(history)
        content, stop_reason = await stream_turn(tools, history)
        history.append({"role": "assistant", "content": content})
        if stop_reason != "tool_use":
            return
        results = []
        for block in content:
            if block["type"] == "tool_use":
                session, tool = routes[block["name"]]
                out = await session.call_tool(tool, block["input"])
                text = "\n".join(c.text for c in out.content if c.type == "text")
                results.append({"type": "tool_result", "tool_use_id": block["id"], "content": text})
        history.append({"role": "user", "content": results})


async def main():
    async with AsyncExitStack() as stack:
        sessions = {}
        for name, cmd in SERVERS.items():
            params = StdioServerParameters(command=cmd[0], args=cmd[1:])
            read, write = await stack.enter_async_context(stdio_client(params))
            sessions[name] = await stack.enter_async_context(ClientSession(read, write))
            await sessions[name].initialize()
        history = []
        while True:
            await chat_turn(sessions, history, input("\n> "))


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        mcp.run()
    else:
        asyncio.run(main())
