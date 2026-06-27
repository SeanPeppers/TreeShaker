# TreeShaker — project rules

## Keep docs and the MCP server in sync with the code

Whenever a change touches the public API surface (`Config` or any nested
config dataclass in `src/treeshaker/config.py`, `FeatureSelectionPipeline`'s
public methods/attributes in `src/treeshaker/pipeline.py`, or CLI flags in
`src/treeshaker/cli.py`), update BOTH of these in the same PR:

1. **README.md** — keep the CLI flag reference table and feature-section
   descriptions accurate. Don't claim behavior that isn't actually in the
   code; don't leave stale flags/fields documented after a rename/removal.
2. **`src/treeshaker/mcp_server.py`** (the local MCP server) — if a new
   config field, pipeline output, or dtype-handling behavior is added,
   reflect it in `list_config_schema()` and the relevant tool's return
   shape. The MCP server should never silently drift from what the
   Python API actually supports — a client introspecting it should get a
   truthful answer.

This is a hard requirement, not a nice-to-have: a PR that changes the API
without updating both is incomplete. If you're a subagent picking up a
TreeShaker task, read this file first.

## One PR per fix or feature

Each bug fix or feature gets its own branch/PR. Don't bundle multiple
unrelated fixes into one PR, even if found in the same testing session.

## Validate against real data, not just synthetic fixtures

Synthetic test fixtures are necessary for CI but not sufficient. Before
trusting a dtype-handling, leakage-detection, or harm-detection change,
smoke-test it against a real public dataset (e.g. via
`sklearn.datasets.fetch_openml`) that's messier than what the existing
test suite covers — real data surfaces edge cases synthetic fixtures
don't (see: the SHAP additivity crash on Ames housing, the categorical
leakage gap on Titanic's `boat` column, and the harm-detection
drop-everything bug on the credit-fraud dataset — all found this way,
none caught by synthetic tests alone).
