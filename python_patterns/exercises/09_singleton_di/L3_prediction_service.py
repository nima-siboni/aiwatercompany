"""
PR #293: Request handlers for the uptake-prediction service

Framework-agnostic handlers for our internal prediction API. The platform server
turns each HTTP request into a dict (`{"headers": {...}, "json": {...}}`), calls
a `handle_*` function from its worker threads and serialises the dict we return.
Users send structure IDs plus adsorption conditions and get predicted CO2
uptakes back; the predictions are saved under a job ID so they can be fetched
again later. At start-up the server wires in the real auth client, results store
and model class with `configure()`, then calls `warm_up()` so that the first
request doesn't pay for loading the model.
"""
import functools
import itertools
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

MODEL_WEIGHTS = "/models/uptake-gnn/v7.pt"
DEFAULT_CONDITIONS = {"temperature_k": 298.0, "pressure_bar": 1.0}

CURRENT_USER: "User | None" = None
METRICS = {"requests": 0, "errors": 0, "seconds": 0.0}

_verify_token: Callable[[str], "User"] | None = None
_store: Any = None
_model_factory: Callable[[], Any] | None = None
_model: Any = None
_model_lock = threading.Lock()
_job_ids = itertools.count(1)


@dataclass(frozen=True)
class User:
    name: str
    projects: frozenset[str]


def configure(
    *, verify_token: Callable[[str], User], store: Any, model_factory: Callable[[], Any]
) -> None:
    """Wire in the auth client, the results store and the model class."""
    global _verify_token, _store, _model_factory
    _verify_token = verify_token
    _store = store
    _model_factory = model_factory


def get_model() -> Any:
    """The surrogate model, loaded once per process and shared by all requests."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = _model_factory()
                _model.load_weights(MODEL_WEIGHTS)
    return _model


def warm_up() -> None:
    get_model()
    _store.ping()


@functools.cache
def _load_structure(structure_id: str) -> Any:
    """Fetch and parse a structure. Parsing the CIF is the slow part, so keep it."""
    return _store.get_structure(structure_id)


def _authenticate(request: dict) -> None:
    global CURRENT_USER
    header = request.get("headers", {}).get("Authorization", "")
    if header.startswith("Bearer "):
        CURRENT_USER = _verify_token(header.removeprefix("Bearer "))


def _require_project(project: str) -> None:
    if CURRENT_USER is None or project not in CURRENT_USER.projects:
        raise PermissionError(f"no access to project {project!r}")


def _endpoint(handler: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Map exceptions to HTTP statuses and record request metrics."""

    @functools.wraps(handler)
    def wrapper(request: dict) -> dict:
        started = time.perf_counter()
        ok = False
        try:
            body = handler(request)
            ok = True
            return {"status": 200, "body": body}
        except PermissionError as exc:
            return {"status": 403, "body": {"error": str(exc)}}
        except (KeyError, ValueError) as exc:
            return {"status": 400, "body": {"error": f"bad request: {exc}"}}
        finally:
            METRICS["requests"] += 1
            METRICS["seconds"] += time.perf_counter() - started
            if not ok:
                METRICS["errors"] += 1

    return wrapper


@_endpoint
def handle_predict(request: dict) -> dict:
    _authenticate(request)
    body = request["json"]
    project = body["project"]
    _require_project(project)

    conditions = DEFAULT_CONDITIONS
    conditions.update(body.get("conditions", {}))

    model = get_model()
    uptakes = {}
    for structure_id in body["structure_ids"]:
        structure = _load_structure(structure_id)
        uptakes[structure_id] = float(model.predict([structure], **conditions)[0])

    job_id = f"pred-{next(_job_ids):06d}"
    _store.save_predictions(job_id, project=project, conditions=conditions, uptakes=uptakes)
    return {"job_id": job_id, "conditions": conditions, "uptake_mmol_g": uptakes}


@_endpoint
def handle_get_results(request: dict) -> dict:
    _authenticate(request)
    job_id = request["json"]["job_id"]
    record = _store.load_predictions(job_id)
    _require_project(record["project"])
    return record


def handle_metrics(request: dict) -> dict:
    """Unauthenticated: scraped by the monitoring stack."""
    return {"status": 200, "body": dict(METRICS)}


def reset_for_tests() -> None:
    global CURRENT_USER, _model
    CURRENT_USER = None
    _model = None
    METRICS.update(requests=0, errors=0, seconds=0.0)
