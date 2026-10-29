"""
PR #210: Benchmark harness for the materials research agent

Runs the agent on our 40-question benchmark, grades each answer with an LLM
judge, and reports accuracy and cost so we can compare prompt and model
changes before shipping them.
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor

import anthropic

from agent import Agent, SqliteMemory  # Agent.run(question) -> Result(answer, usage)

client = anthropic.Anthropic()
JUDGE_MODEL = "claude-opus-5-5"
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}

with open("benchmark.jsonl") as f:
    BENCHMARK = [json.loads(line) for line in f]

FEW_SHOT = random.sample(BENCHMARK, 3)
SYSTEM_PROMPT = "You are a materials research assistant.\n\nExamples:\n" + "\n\n".join(
    f"Q: {ex['question']}\nA: {ex['answer']}" for ex in FEW_SHOT
)

agent = Agent(system_prompt=SYSTEM_PROMPT, memory=SqliteMemory("agent_memory.db"))

JUDGE_PROMPT = """Here is a question, a reference answer, and a candidate answer.
Rate the candidate's correctness from 1 to 10. Reply with the number only.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}"""


def judge(question, reference, candidate):
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question, reference=reference, candidate=candidate
            ),
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return int(text.strip()[0])


def run_one(example):
    result = agent.run(example["question"])
    score = judge(example["question"], example["answer"], result.answer)
    return {"id": example["id"], "score": score, "usage": result.usage}


def main():
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_one, ex) for ex in BENCHMARK]
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception:
                continue

    accuracy = sum(r["score"] >= 7 for r in results) / len(results)
    cost = sum(
        r["usage"].input_tokens / 1e6 * PRICE_PER_MTOK["input"]
        + r["usage"].output_tokens / 1e6 * PRICE_PER_MTOK["output"]
        for r in results
    )
    print(f"accuracy: {accuracy:.1%}   cost: ${cost:.2f}")


if __name__ == "__main__":
    main()
