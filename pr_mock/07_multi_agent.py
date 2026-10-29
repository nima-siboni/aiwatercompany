"""
PR #201: Planner / worker / critic loop for the MOF design campaign

The planner proposes candidate MOFs for a design brief, workers evaluate
each candidate in parallel (surrogate-model predictions plus the LLM's
judgement), and a critic reviews the shortlist. We iterate until the critic
approves.
"""
import asyncio
import json

import anthropic

from surrogate import predict_properties  # GNN surrogate: MOF name -> dict of predicted properties

client = anthropic.Anthropic()
MODEL = "claude-opus-5-5"

PLANNER_PROMPT = (
    "You are a MOF design expert. Given a design brief, propose candidate MOFs. "
    "Respond with a JSON list of objects with keys 'name' and 'rationale'."
) # it is not clear what is name supposed to be.
WORKER_PROMPT = (
    "You evaluate a single MOF candidate. Using the predicted properties provided, "
    "give a score from 0 to 10 and a short justification."
)
CRITIC_PROMPT = (
    "You review a shortlist of MOF candidates. If the shortlist is good, reply APPROVED. "
    "Otherwise explain what is wrong so the planner can improve it."
)


def ask(system, messages):
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=messages,
    )
    return "".join(b.text for b in response.content if b.type == "text")


async def evaluate_candidate(candidate, history):
    props = predict_properties(candidate["name"])
    history.append({
        "role": "user",
        "content": f"Candidate: {candidate['name']}\nPredicted properties: {json.dumps(props)}",
    })
    reply = ask(WORKER_PROMPT, history)
    history.append({"role": "assistant", "content": reply})
    return {"name": candidate["name"], "props": props, "evaluation": reply}


async def run_campaign(brief):
    planner_history = [{"role": "user", "content": brief}] # [nit] a better name is designer
    worker_history = [] # [nit] wrker is a very general concept, we can be way more specific.
    round_num = 0

    while True: # [Blocker] unbounded while loop, might never end leading to waste of resources, why not limiting the campaign to some budgets (time, token, canidates, etc.)
        round_num += 1
        plan = ask(PLANNER_PROMPT, planner_history)
        planner_history.append({"role": "assistant", "content": plan})
        candidates = json.loads(plan)

        evaluations = await asyncio.gather(
            *(evaluate_candidate(c, worker_history) for c in candidates)
        ) # would this setup allow allocating the resources to other candidates as time

        shortlist = sorted(evaluations, key=lambda e: e["evaluation"], reverse=True)[:5]
        summary = "\n".join(f"- {e['name']}: {e['evaluation'][:200]}" for e in shortlist)
        verdict = ask(CRITIC_PROMPT, [{"role": "user", "content": summary}])

        if "APPROVED" in verdict.upper():
            return shortlist

        planner_history.append({
            "role": "user",
            "content": (
                f"Round {round_num} critic feedback:\n{verdict}\n\n"
                f"All evaluations so far:\n{json.dumps(evaluations)}"
            ),
        })
