# Contributing

## Getting set up

See the [README](README.md#setup) for full setup instructions (local and
Databricks). In short:

```bash
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python src/utils/generate_sample_data.py --rows 50000 --seed 42
pytest
```

Windows: Spark needs `HADOOP_HOME`/`winutils.exe` set up even for local
runs — see the README's setup section before anything else.

## Conventions this codebase follows

Consistency matters more than any individual rule here — before adding a
new stage or module, skim a sibling file (`src/transformation/transform.py`
and `src/ml/feature_engineering.py` are good examples) and match its shape
rather than introducing a new one.

- **Config-driven, not hardcoded.** Every stage reads its settings from
  `config/config.yaml` via a `load_run_settings(config_path, env_override)`
  function returning a frozen `@dataclass`. Add new config there rather
  than as a module-level constant or a new CLI-only flag.
- **`--config` / `--env` on every CLI entry point**, resolved through
  `src/utils/config.py`'s `resolve_environment`, so every stage works
  identically under `environment: local` and `environment: databricks`
  without code changes.
- **Docstrings explain *why*, not *what*.** A function name and its
  arguments already say what it does; a docstring or inline comment earns
  its place by explaining a non-obvious constraint, a rejected alternative,
  or a performance/correctness tradeoff. If you'd delete a comment and lose
  nothing, don't add it in the first place.
- **Structured JSON logging**, not print statements, via
  `src/utils/logging_utils.py`'s `setup_logging()` + `logging.getLogger(__name__)`.
  Never pass `extra={"name": ...}` (or any other `logging.LogRecord`
  attribute name) — it collides with the record's own fields and raises at
  runtime; this bit us once already (see `sql_runner.py`'s git history).
- **Pure functions over classes where the state is just "a DataFrame
  becomes another DataFrame."** Most transformation logic here is a
  function taking a `DataFrame` (plus a couple of plain parameters) and
  returning a `DataFrame` — easy to unit test with a small mock DataFrame,
  easy to call from either the CLI (`main()`) or the Databricks notebook
  without re-deriving anything.
- **Every optimization decision gets a comment saying why**, in enough
  detail that someone could explain it in an interview without re-deriving
  it from scratch. See `transform.py` for the density this project aims
  for — broadcast joins, repartition choices, and caching all say *why*,
  not just *that*.

## Before opening a PR

1. **Run the pipeline against the sample data** for any stage you touched —
   `python src/pipeline.py --stage <your-stage>` — not just `pytest`. The
   test suite covers logic on small mock DataFrames; it won't catch
   everything a real run would (see `sql/query_optimization_notes.md` and
   the README's engineering-decisions section for examples of bugs that
   were only caught this way).
2. **Run the test suite**: `pytest`. Add tests alongside new logic — see
   `tests/factories.py` for the mock-DataFrame helpers before writing a new
   one from scratch.
3. **If you touched a Spark transformation**, sanity-check the physical
   plan for anything you claim in a comment (a broadcast join, an elided
   shuffle) with `df.explain(mode="formatted")` rather than asserting it
   from memory — `src/transformation/sql_runner.py`'s
   `--compare-optimizations` flag is a working example of capturing and
   documenting a real plan instead of an assumed one.
4. **If you changed `config.yaml`'s schema**, update every
   `load_run_settings` that reads it and the corresponding section in the
   README/notebook, since several files read the same keys independently
   (see `CONTRIBUTING.md`'s config-driven-not-hardcoded note above — that
   consistency has to be maintained by hand across files, there's no single
   schema definition enforcing it yet).

## Adding a new pipeline stage

1. Add its settings to `config.yaml` and a matching `@dataclass` +
   `load_run_settings` in the new module.
2. Write the transformation logic as plain functions taking/returning
   `DataFrame` (or `pandas.DataFrame` for the ML stages), separate from the
   CLI plumbing (`main()`, `parse_args()`), so it stays directly reusable
   from `notebooks/databricks_pipeline.py` and from tests.
3. Register it in `src/pipeline.py`'s `STAGES` dict and `STAGE_ORDER` list.
4. Add the equivalent cells to `notebooks/databricks_pipeline.py`, calling
   the same functions directly (no `build_spark_session()`, no subprocess —
   see that file's own architecture notes).
5. Add unit tests in `tests/`, using or extending `tests/factories.py`.
