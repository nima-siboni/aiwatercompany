# Answer key: L3_engine_plugins.py — Factory and registry, level 3 (35 min + 5 min summary)

**Pattern focus:** A plugin system combines three risky things: importing code by name, registering classes automatically, and caching instances. This drill trains you to ask who controls the names, what the registry key is derived from, and whether cached objects carry per-request state.

**Good clarifying questions:**
- Who can call `POST /jobs`: only our own services, or any authenticated user? Where does `"plugin"` come from in practice? This decides whether #1 is a blocker or a hardening item. Assume API users can set it.
- Does the service handle requests on several threads (a typical WSGI/ASGI worker pool)? (#6)
- How is a plugin meant to be installed: as a package in the service image, or as something teams upload?
- What actually makes an engine expensive to create? The docstring says force-field reads. Is there anything else (licences, connections)?

## Blockers (silently wrong results / data loss / security)
1. **Any API caller can make the service import any module.** `load_plugin(request["plugin"])` calls `importlib.import_module` on a string from the request. Importing a module runs its top-level code inside the service, with the service's credentials. So every importable module becomes something a caller can trigger: a maintenance script on `sys.path` whose top level deletes or migrates data, a test helper, or a file someone managed to put in a directory on `sys.path` (a shared workspace, the working directory). It's also the way in for #4: a module that defines a class called `GCMCEngine` replaces the production engine for every user. *Fix:* requests name only the engine. Plugins come from deployment config (an allowlist of module paths, or entry points) and are loaded at start-up, and `load_plugin` refuses anything else.

2. **`get_engine` caches by name only, so the first request's params win for everyone.** `_instances` is keyed by `name`, and `params` are used only when the engine is first built. If the first `gcmc` request after a restart is a quick pre-screen with `{"cycles": 2000}`, every later production request that asks for 50000 silently gets 2000-cycle input files, and those noisy uptakes end up in the training data. Which params win depends on which request happened to arrive first. *Fix:* don't cache engines. Construction is cheap once the force field is cached (#7). If you must cache, key on the name plus the frozen params, and bound the cache.

3. **`input_files` writes each job's conditions into the shared engine.** `cond = self.conditions` is an alias, not a copy, so `cond.update(conditions)` changes the engine's defaults. A job at `temperature=77` makes 77 K the default for every later `gcmc` job that doesn't set a temperature, including other users' jobs, because of the shared engine in #2. Their results are labelled 298 K. *Fix:* `cond = {**self.conditions, **conditions}`, and treat engines as immutable after construction.

4. **Registry names come from `cls.__name__`, and a later class silently replaces an earlier one.** `ENGINES[cls.__name__.lower().removesuffix("engine")] = cls` overwrites without checking. Any class named `GCMCEngine` anywhere (a fork in another team's plugin, a test double, a module imported through #1) takes over `"gcmc"` for every request, depending on import order. Renaming a class also silently renames the public engine name, which breaks stored job specs. *Fix:* register under an explicit name passed as a class keyword (`class GCMCEngine(RaspaEngine, name="gcmc")`), and raise on duplicates.

## Major
5. **The shared base class is registered and offered as an engine.** `__init_subclass__` runs for every subclass, including `RaspaEngine`, so `available_engines()` returns `["gcmc", "raspa", "widom"]`, and the web form built from `describe_engines()` offers `raspa` too. A user who picks it gets `TypeError: Can't instantiate abstract class RaspaEngine without an implementation for abstract method 'prepare'`. *Fix:* with explicit names (#4), shared base classes just don't pass one. Checking `inspect.isabstract(cls)` inside `__init_subclass__` also works on Python 3.13 (I checked: `inspect` handles a class that's still being built), but it doesn't fix #4.

6. **Lazy creation and shared engines aren't thread-safe.** In a threaded server, two first requests can both see `_instances.get(name)` return `None`, both build an engine, and the last write wins. That's mostly wasted work. Combined with #3 it's worse: concurrent `prepare` calls on one engine update the same `self.conditions` dict, so two simultaneous jobs that both set a temperature can each be written with the other's. *Fix:* removing the shared mutable engine (#2, #3) removes the race. If you keep a cache, guard creation with a lock and keep cached objects immutable.

7. **The engine cache doesn't save what the docstring says it saves.** `mixing_rules` is a property that re-reads and re-parses `FORCEFIELD_DIR / "UFF.def"` on every `prepare` call, so a 10,000-job screen still does 10,000 reads from the shared filesystem. The cache brings the risks in #2, #3 and #6 and none of the benefit. *Fix:* cache the parsed force field by path (`functools.lru_cache` on a module-level function that returns an immutable string or tuple), and build a fresh, cheap engine per request.

## Minor
8. **The force-field parser crashes on comments and blank lines.** `atom_type, epsilon, sigma = line.split()` raises `ValueError` (too many, or not enough, values to unpack) for a header such as `# UFF, epsilon in K` or an empty line. Hand-maintained force-field files often have both. *Fix:* skip blank and `#` lines, and say which file and line failed.

9. **An unknown engine is a bare `KeyError`.** `ENGINES[name]` raises `KeyError: 'gmc'`, which most web frameworks turn into a 500. It doesn't say what's available, or that the plugin providing the engine may not be loaded. *Fix:* `ValueError(f"unknown engine {name!r}; available: {available_engines()}")`, mapped to a 400.

10. **Request strings go into input files and paths unvalidated.** `structure_id` is interpolated into `simulation.input`, so `"mof-1\nNumberOfCycles 999999999"` adds a directive of the caller's choosing to the RASPA input. The same newline forges a line in the service log. `params["forcefield"]` becomes part of a path, so `"../../home/alice/notes"` reads any `.def` file the service can see, and echoes it in the response if it parses. It's Minor because it's limited to `.def` files and to internal callers; if the API is exposed more widely, upgrade it. *Fix:* validate `structure_id` and the force-field name against a strict pattern (or a list of existing files) at the API boundary.

11. **Module-level mutable registries make tests order-dependent.** `ENGINES` and `_instances` live for the whole process. A test that defines an engine leaks it into every later test, and there's no way to reset them. *Fix:* a small registry object owned by the service, created fresh in tests. This is a design change, so raise it as a follow-up rather than blocking on it.

## Red herrings (say why they're fine)
- **Calling `importlib.import_module` on every request.** It looks expensive, but after the first import it's a `sys.modules` lookup, and the plugin's code runs once per process (I checked with a module that counts its own imports). The problem with that line is trust (#1), not speed.
- **`DEFAULT_CONDITIONS` is a mutable class attribute.** That's the classic shared-state smell, but `__init__` copies it with `dict(self.DEFAULT_CONDITIONS)` and nothing mutates the class dict. The leak in #3 goes through `self.conditions`, not the class attribute.

## Model 1-minute summary
"First, a question: can API users set `"plugin"`? If so, any caller can make us import any module, which runs code with the service's credentials. And because registration is by class name, that module can also replace the production `gcmc` engine for everyone. Plugins should come from deployment config, register under explicit names, and duplicates should be rejected. The second group is silent wrong results from caching engines by name. The first request's parameters, say 2000 cycles, are used for every later request, and each job's conditions are written into the shared engine, so a 77 K job changes the default temperature for everyone after it. The cache doesn't even save the force-field reads it was added for. I'd drop it, cache the parsed force field instead, and build an engine per request, which also removes the thread-safety problem. Smaller items: the abstract base offered as an engine, and validating `structure_id`."

## Reference fix
`L3_engine_plugins_fixed.py`:
- `load_plugin` only imports modules on `ALLOWED_PLUGINS`, which comes from deployment config.
- Engines register with an explicit class keyword (`name="gcmc"`), and duplicates raise. `RaspaEngine` has no name, so it isn't offered.
- There's no engine cache: `get_engine` builds a fresh engine per request, and `input_files` merges conditions into a new dict. The parsed force field is cached by path with `functools.lru_cache`, and the parser skips comments and blank lines.
- `validate_request` checks `structure_id`, `RaspaEngine` checks the force-field name, and an unknown engine raises a `ValueError` that lists the available ones.
- #11 (module-level registries) is left as it is, to keep the drill's API.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_05_factory_registry.py -k L3 -v`
