"""
PR #193: Tools for the literature-review agent

Adds the tool implementations the agent uses to pull papers, search our
materials API, run quick analyses, keep notes in the shared workspace, and
email summaries to the team. The agent loop itself is unchanged (agent.py
calls handle_tool_call for every tool_use block).
"""
import contextlib
import io
import os
import smtplib
import traceback
from email.message import EmailMessage

import requests

WORKSPACE = "/srv/agent-workspace"
MATERIALS_API = "https://materials-api.internal/v1/search"
MATERIALS_API_KEY = os.environ["MATERIALS_API_KEY"]


def fetch_url(url):
    """Download a paper or web page and return its text."""
    resp = requests.get(url) # This is going to fail badly! I htink we cannot use a general fetch and expect a readable .text from it, lets say a pdf is returned where the pdf is not machine generated, it is just scan of a couple of pages.
    return resp.text


def search_materials(query, max_results=20, **kwargs): # what happens to **kwargs?
    """Search the internal materials API."""
    resp = requests.get(
        f"{MATERIALS_API}?q={query}&limit={max_results}&api_key={MATERIALS_API_KEY}",
        timeout=10,
    ) # no error handling?
    resp.raise_for_status()
    return resp.json()


def read_file(path): # error handling is missing.
    # also if the file does not exist it creates it? wht is the open mode?
    with open(os.path.join(WORKSPACE, path)) as f:
        return f.read()


def write_file(path, content):
    full = os.path.join(WORKSPACE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"


def run_python(code):
    """Run analysis code and return whatever it prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, globals())
    return buf.getvalue()


def email_summary(to, subject, body):
    msg = EmailMessage()
    msg["From"] = "lit-agent@aiwatercompany.com"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP("smtp.internal") as s:
        s.send_message(msg)
    return "sent"


TOOL_FUNCS = {
    "fetch_url": fetch_url,
    "search_materials": search_materials,
    "read_file": read_file,
    "write_file": write_file,
    "run_python": run_python,
    "email_summary": email_summary,
}

TOOLS = [
    {
        "name": "fetch_url",
        "description": "Fetch a URL.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "search_materials",
        "description": "Search for materials.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Maximum number of results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the shared workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Save notes or results to the shared workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_python",
        "description": "Run Python code for quick data analysis. numpy and pandas are installed.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "email_summary",
        "description": "Email a summary of findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


def handle_tool_call(block):
    try:
        result = TOOL_FUNCS[block.name](**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": result}
    except Exception:
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": traceback.format_exc(),
            "is_error": True,
        }
