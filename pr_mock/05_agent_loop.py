"""
PR #188: Materials research agent (tool-use loop)

Lets a chemist ask things like "find water-stable MOFs with CO2 uptake above
4 mmol/g and queue GCMC runs for the top 3". The agent can query our
materials database and submit simulations to the cluster.
"""
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import anthropic

from cluster import submit_gcmc_job  # talks to the scheduler, returns a job id string


# the whole code is full of magic numbers!
client = anthropic.Anthropic()
MODEL = "claude-opus-5-5"
DB_PATH = "materials.db"

TOOLS = [ # In the tools definition dont we specify the shape of the output?
    {
        "name": "query_materials",
        "description": (
            "Run a SQL query against the materials database. Table `mofs` has columns: "
            "mof_id, metal, linker, pore_diameter, surface_area, co2_uptake, water_stable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "SQL query to run"}},
            "required": ["sql"], # [NIT] rename sql -> sql_query
        },
    },
    {
        "name": "submit_simulation",
        "description": "Submit a GCMC adsorption simulation for one MOF. Returns a job id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mof_id": {"type": "string"},
                "temperature_k": {"type": "number"},
                "pressure_bar": {"type": "number"},
            },
            "required": ["mof_id", "temperature_k", "pressure_bar"],
        },
    },
]


def system_prompt():
    return (
        "You are a materials-discovery assistant for the CO2 capture team. "
        "Use the tools to answer questions and queue simulations. "
        f"Current time: {datetime.now().isoformat()}" # what the fuck is this?
    )


def query_materials(sql):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(sql).fetchall()
    conn.close()
    return json.dumps(rows)


def submit_simulation(mof_id, temperature_k, pressure_bar): # [Minor] highly recommneded to have a dataclass or even better a pydantic model.
    for attempt in range(3):
        try:
            return submit_gcmc_job(mof_id, temperature_k, pressure_bar, timeout=30) # [Blocker] this is not the signature of the submit job.
        except TimeoutError: # [Major] in every cluster there is a queying system.
            # [Major] Other errors are swallowed.
            time.sleep(2) 
    raise RuntimeError("cluster unavailable")


def execute_tool(name, tool_input):
    try:
        if name == "query_materials":
            return query_materials(tool_input["sql"])
        if name == "submit_simulation":
            return submit_simulation(**tool_input)
    except Exception:
        return "[]" # [Major] Tool execution;s errors are swallowed. 


def call_model(messages):
    for attempt in range(5):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=system_prompt(),
                tools=TOOLS,
                messages=messages,
                cache_control={"type": "ephemeral"},
            ) # Multiple retries could lead to token limits per sec. or per min.
        except Exception as e:
            print(f"API error, retrying: {e}")
            time.sleep(1)
    raise RuntimeError("model unavailable") # [Major] make the error message clear.


def run_agent(question, messages=[]): # What will happen if we run this twice?
    messages.append({"role": "user", "content": question})

    while True:
        response = call_model(messages)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return response.content[0].text

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(execute_tool, b.name, b.input) for b in tool_blocks]
            tool_results = [
                {"type": "tool_result", "tool_use_id": block.id, "content": fut.result()}
                for block, fut in zip(tool_blocks, as_completed(futures))
            ] # What will happen if one of the tools fail?
        messages.append({"role": "user", "content": tool_results})
# [Blocking] MEssages are never returned?! How do we get something out of this loop?
# [Blocking] There is nothing to take care of resource management in this loop! What will happen when we get close to context limit?
# [Blocking] the format of mof_id is not clear. The temperature and pressure units are not clear. All of this needs to be clarified in the description, and best would be to check the values before submitting to the tools. There is no point in running experiiments in -1K! ;)
# The foramt of the output is alos not clear, for neither of the tools.