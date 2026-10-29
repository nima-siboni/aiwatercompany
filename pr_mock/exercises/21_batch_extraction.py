"""
PR #262: Extract CO2 uptake, BET area and heat of adsorption from 20k MOF papers

Runs every paper in the corpus (text already extracted from the PDFs) through Claude Haiku 5.5
with a JSON schema, using the Message Batches API at half price. Uptakes are converted to
mmol/g and written as one row per paper to adsorption_db.csv, which feeds the training set of
the uptake surrogate. To filter out hallucinations, a row is kept only if all of its numbers
appear in the paper text.
"""
import json
import time

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"

INSTRUCTIONS = (
    "You extract adsorption data from papers about metal-organic frameworks. Report the main "
    "MOF of the paper, its CO2 uptake at 298 K and 1 bar in the unit the paper uses, its BET "
    "surface area in m2/g and its isosteric heat of adsorption of CO2 in kJ/mol."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "mof_name": {"type": "string"},
        "co2_uptake": {"type": "number"},
        "co2_uptake_unit": {"type": "string"},
        "bet_m2_g": {"type": "number"},
        "qst_kj_mol": {"type": "number"},
    },
    "required": ["mof_name", "co2_uptake", "co2_uptake_unit", "bet_m2_g", "qst_kj_mol"],
    "additionalProperties": False,
}

TO_MMOL_G = {"mmol/g": 1.0, "mol/kg": 1.0, "cm3/g": 1 / 22.414, "mg/g": 1 / 44.01, "wt%": 10 / 44.01}


def make_request(i, paper):
    return Request(
        custom_id=f"paper-{i}",
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=1024,
            system=INSTRUCTIONS,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": paper["text"], "cache_control": {"type": "ephemeral"}},
            ]}],
        ),
    )


def run_batch(papers):
    batch = client.messages.batches.create(requests=[make_request(i, p) for i, p in enumerate(papers)])
    while batch.processing_status != "ended":
        time.sleep(60)
        batch = client.messages.batches.retrieve(batch.id)
    return client.messages.batches.results(batch.id)


def to_mmol_g(value, unit):
    return value * TO_MMOL_G.get(unit.strip().lower(), 1.0)


def grounded(record, text):
    numbers = [record["co2_uptake"], record["bet_m2_g"], record["qst_kj_mol"]]
    return all(str(v) in text for v in numbers)


def extract(papers):
    rows = []
    for paper, result in zip(papers, run_batch(papers)):
        if result.result.type != "succeeded":
            continue
        message = result.result.message
        text = next(b.text for b in message.content if b.type == "text")
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not grounded(record, paper["text"]):
            continue
        rows.append({
            "doi": paper["doi"],
            "mof": record["mof_name"],
            "co2_mmol_g": to_mmol_g(record["co2_uptake"], record["co2_uptake_unit"]),
            "bet_m2_g": record["bet_m2_g"],
            "qst_kj_mol": record["qst_kj_mol"],
        })
    pd.DataFrame(rows).to_csv("adsorption_db.csv", index=False)
    return rows


if __name__ == "__main__":
    corpus = pd.read_parquet("papers_text.parquet").to_dict("records")  # doi, text
    rows = extract(corpus)
    print(f"extracted {len(rows)} records")
