"""
Reference fix for PR #293 (python_patterns/exercises/09_singleton_di/L3_prediction_service.py).
See the answer key for the reasoning.
"""
import functools
import threading
import time
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

MODEL_WEIGHTS = "/models/uptake-gnn/v7.pt"
DEFAULT_CONDITIONS = MappingProxyType({"temperature_k": 298.0, "pressure_bar": 1.0})
MAX_STRUCTURES_PER_REQUEST = 1_000

METRICS = {"requests": 0, "errors": 0, "seconds": 0.0}
_metrics_lock = threading.Lock()

_verify_token: Callable[[str], "User"] | None = None
_store: Any = None
_model_factory: Callable[[], Any] | None = None
_model: Any = None
_model_lock = threading.Lock()


@dataclass(frozen=True)
class User:
    name: str
    projects: frozenset[str]


def configure(
    *, verify_token: Callable[[str], User], store: Any, model_factory: Callable[[], Any]
) -> None:
    """Wire in the auth client, the results store and the model class.

    Anything built from the previous dependencies (the loaded model, cached
    structures) is dropped, so tests and reconfiguration don't see stale state.
    """
    global _verify_token, _store, _model_factory, _model
    with _model_lock:
        _verify_token, _store, _model_factory = verify_token, store, model_factory
        _model = None
    _load_structure.cache_clear()


def get_model() -> Any:
    """The surrogate model, loaded once per process and shared by all requests."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model = _model_factory()
                model.load_weights(MODEL_WEIGHTS)
                _model = model  # publish only once it is fully loaded
    return _model


def warm_up() -> None:
    """Load the model. Run it in each worker process; the store connects lazily there."""
    get_model()


@functools.lru_cache(maxsize=4096)
def _load_structure(structure_id: str) -> Any:
    """Fetch and parse a structure. Parsing the CIF is the slow part, so keep recent ones."""
    return _store.get_structure(structure_id)


def _authenticate(request: dict) -> User:
    header = request.get("headers", {}).get("Authorization", "")
    if not header.startswith("Bearer "):
        raise PermissionError("missing bearer token")
    return _verify_token(header.removeprefix("Bearer "))


def _require_project(user: User, project: str) -> None:
    if project not in user.projects:
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
            elapsed = time.perf_counter() - started
            with _metrics_lock:
                METRICS["requests"] += 1
                METRICS["seconds"] += elapsed
                if not ok:
                    METRICS["errors"] += 1

    return wrapper


@_endpoint
def handle_predict(request: dict) -> dict:
    user = _authenticate(request)
    body = request["json"]
    project = body["project"]
    _require_project(user, project)

    given = body.get("conditions", {})
    unknown = given.keys() - DEFAULT_CONDITIONS.keys()
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    conditions = {**DEFAULT_CONDITIONS, **given}  # a new dict for this request

    structure_ids = list(body["structure_ids"])
    if len(structure_ids) > MAX_STRUCTURES_PER_REQUEST:
        raise ValueError(f"at most {MAX_STRUCTURES_PER_REQUEST} structures per request")

    structures = [_load_structure(sid) for sid in structure_ids]
    predicted = get_model().predict(structures, **conditions)  # one batched call
    uptakes = {sid: float(u) for sid, u in zip(structure_ids, predicted, strict=True)}

    job_id = f"pred-{uuid.uuid4().hex}"
    _store.save_predictions(job_id, project=project, conditions=conditions, uptakes=uptakes)
    return {"job_id": job_id, "conditions": conditions, "uptake_mmol_g": uptakes}


@_endpoint
def handle_get_results(request: dict) -> dict:
    user = _authenticate(request)
    job_id = request["json"]["job_id"]
    record = _store.load_predictions(job_id)
    _require_project(user, record["project"])
    return record


def handle_metrics(request: dict) -> dict:
    """Unauthenticated: scraped by the monitoring stack."""
    return {"status": 200, "body": dict(METRICS)}
