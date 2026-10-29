"""
PR #287: Upsert DFT results into Postgres and export new ones nightly for MLIP training

`ingest()` is called by the workflow manager whenever a batch of DFT single points finishes. It
upserts them into `calculations`, keyed by (structure, functional, Hubbard U), so a recomputed
structure replaces its old row. `export_new()` runs nightly: it reads every calculation updated
since the last export, skips structures already in the training set (by fingerprint), and
writes the rest to data/train/ as Parquet, partitioned by functional. The MLIP training job
reads the whole directory with `load_training_set()`.
"""
import json
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class Calculation(Base):
    __tablename__ = "calculations"
    id = Column(Integer, primary_key=True)
    structure_id = Column(String, nullable=False)
    functional = Column(String, nullable=False)
    u_value = Column(Float)  # Hubbard U in eV, NULL when the run has no +U
    cif = Column(Text, nullable=False)
    energy_ev = Column(Float, nullable=False)
    forces = Column(Text, nullable=False)  # JSON, [[fx, fy, fz], ...] in eV/Å
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint("structure_id", "functional", "u_value"),)


engine = create_engine("postgresql+psycopg://workflow@db/results")
ROOT = "data/train"
WATERMARK = "data/train_last_export.json"


def ingest(records):
    """records: dicts with structure_id, functional, u_value, cif, energy_ev, forces."""
    stmt = insert(Calculation).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["structure_id", "functional", "u_value"],
        set_={"cif": stmt.excluded.cif, "energy_ev": stmt.excluded.energy_ev,
              "forces": stmt.excluded.forces},
    )
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()


def read_watermark():
    try:
        with open(WATERMARK) as f:
            return datetime.fromisoformat(json.load(f)["updated_at"])
    except FileNotFoundError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def fingerprint(cif):
    return hash(cif)


def load_training_set():
    try:
        return pd.read_parquet(ROOT)
    except FileNotFoundError:
        return None


def export_new():
    since = read_watermark()
    with Session(engine) as session:
        rows = session.scalars(
            select(Calculation).where(Calculation.updated_at > since).order_by(Calculation.updated_at)
        ).all()
    if not rows:
        return 0
    with open(WATERMARK, "w") as f:
        json.dump({"updated_at": rows[-1].updated_at.isoformat()}, f)

    new = pd.DataFrame([{
        "structure_id": r.structure_id, "functional": r.functional, "u_value": r.u_value,
        "fingerprint": fingerprint(r.cif), "energy_ev": r.energy_ev, "forces": r.forces,
    } for r in rows])
    existing = load_training_set()
    if existing is not None:
        new = new[~new.fingerprint.isin(existing.fingerprint)]
    pq.write_to_dataset(pa.Table.from_pandas(new, preserve_index=False), ROOT,
                        partition_cols=["functional"])
    return len(new)
