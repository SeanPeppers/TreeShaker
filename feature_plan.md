# TreeShaker Feature & Infra Plan

Tracks the full backlog from research + infra hardening. Each item below ships as its own branch + PR (except Wave 0, done direct to main). Default-off flags for anything that changes pipeline behavior — existing CLI/API behavior must not change unless explicitly opted into.

---

## Wave 0 — Done (direct to main, no PR)

- [x] README fix: schema (`recommendation`/`reason`), leakage section, CLI flags table — commit `f419605`

---

## Wave -1 — CI/infra hardening (do BEFORE any feature PRs)

Rationale: shipping 18 feature PRs on top of a CI that doesn't even run tests, has no type checking, no pinned deps, no coverage floor, and no dependency CVE scan means every PR is reviewed blind. Harden the gate first so every later PR is actually checked.

| # | Item | What it does | Priority |
|---|---|---|---|
| 1 | **Tests in CI** | Add `pytest` step to `.github/workflows/lint.yml` (or new `test.yml`) — currently only `ruff check`/`ruff format --check` run. Broken code currently ships if lint-clean. | High | **[x] Done** — `test` job in lint.yml |
| 2 | **Type checking** | Add `mypy` (or `pyright`) as dev dep + CI step. ML libs (numpy/pandas/xgboost) drift dtypes/shapes silently — catches that class of bug pre-merge. | High | **[x] Done** — `mypy` job |
| 3 | **Dependency pinning** | Lock exact versions — `uv lock` (preferred, fast) or generate `requirements-lock.txt` via `pip freeze` in a clean venv. `pyproject.toml` deps are all loose `>=` ranges — CI today can silently pick up a breaking upstream release. | High | **[x] Done** — `uv.lock`, `uv sync --locked` in CI |
| 4 | **Coverage threshold** | `pytest-cov` already installed (`pyproject.toml` line 35) but no floor enforced. Add `--cov-fail-under=80` (or measure current baseline first, set realistic floor) to `addopts`. CI fails if coverage drops. | Medium | **[x] Done** — `--cov-fail-under=85`, actual ~90% |
| 5 | **Security scan** | Add `pip-audit` step to CI — one-liner, checks deps against known CVE advisories. | Medium | **[x] Done** — `pip-audit` job, plus gitleaks + zizmor added beyond original scope |
| 6 | **Branch protection** | GitHub → Settings → Branches → `main` → require all CI status checks (ruff, pytest, mypy, pip-audit) green before merge. Manual GitHub setting, not code — do this once infra checks above exist and are green. | Lower (manual, last step of this wave) | **Blocked** — needs GitHub Advanced Security, not available on a private free-tier repo. See `PRE_PUBLIC_CHECKLIST.md`. |

Beyond the original Wave -1 scope, also added: dependabot (with cooldown), pre-commit hooks, PR/issue templates, CODEOWNERS, SECURITY.md, doc-sync CI check, Action SHA-pinning, codecov, PR auto-labeling.

Order: 1 → 2 → 3 → 4 → 5 as separate PRs (small, fast to review, avoids one giant CI-rewrite PR). 6 done manually after 1-5 merged.

---

## Wave 1 — Detection/scoring additions (SHIPPED)

Wave 1 (VIF multicollinearity, SHAP interaction values, mutual information scoring, bootstrap ensemble importance, class imbalance detection, feature groups) — shipped, see CHANGELOG/git history.

## Wave 2 — Alternative algorithms (behind flags, bigger pipeline touch)

| # | Feature | Notes | Status |
|---|---|---|---|
| 7 | BorutaSHAP `--use-boruta` flag | Pending decision from earlier — adds `boruta_decision` column (Confirmed/Tentative/Rejected) | **[x] Done** |
| 8 | RFECV option | Iterative elimination backed by CV score | **[x] Done** |
| 9 | LightGBM backend | Optional model swap to cross-validate importance rankings, faster fit | **[x] Done** |
| 10 | CatBoost backend | Native categorical handling, skip one-hot | **[x] Done** |
| 11 | Random forest baseline importance | Cross-check tree-model ranking against RF | **[x] Done** |
| 12 | Model-agnostic sklearn permutation importance | Support any sklearn estimator, not just XGBoost | **[x] Done** |

## Wave 3 — Performance/scale

| # | Feature | Notes | Status |
|---|---|---|---|
| 13 | GPU acceleration flag | `tree_method=gpu_hist` / RAPIDS path | **[x] Done** — `use_gpu`, CUDA auto-detect |
| 14 | Optuna pruning + early stopping | Skip plateauing trials, cut tuning time | **[x] Done** — `use_optuna_pruning` |
| 15 | Large-dataset sampling strategy | Auto-stratified sampling above a row-count threshold | **[x] Done** — `large_dataset_sample_threshold`/`_frac` |

## Wave 4 — Output/UX & specialized checks

| # | Feature | Notes | Status |
|---|---|---|---|
| 16 | Interactive HTML dashboard (plotly) | Sortable importances, color-coded leakage/recommendation | **[x] Done** — `--format html`, color-coded + embedded SHAP chart |
| 17 | Time-series leakage checks | Look-ahead leakage, temporal autocorrelation dominance | **[x] Done** — `detect_temporal_structure`, `exempt_temporal_features` |
| 18 | MLflow/DVC export | Log rankings + run metadata for experiment tracking | **Not started** |

---

## Wave 5 — Enterprise/scale + research credibility + extensibility

| # | Feature | Notes | Status |
|---|---|---|---|
| 19 | Custom model passthrough | `custom_model` config field — accept any sklearn-API (`.fit`/`.predict`/`.predict_proba`) estimator as primary model instead of hardcoded XGBoost. Covers arbitrary ML/NN models via sklearn-wrapper adapters (skorch, scikeras). | Not started |
| 20 | Async/parallel feature evaluation | Currently single-threaded; slow on 1000+ feature datasets. Parallelize per-feature scoring (joblib/concurrent.futures) where independent. | Not started |
| 21 | Streaming/chunked data support | Datasets too large for RAM — chunked read + incremental fit path. | Not started |
| 22 | Polars/numpy-native path | Optional non-pandas path for performance; pandas stays default. | Not started |
| 23 | Multi-seed stability report | Run full pipeline across N random seeds, report variance/agreement on recommendations (beyond existing single-seed bootstrap). | Not started |
| 24 | Calibrated confidence scores | Replace binary accept/reject recommendation with a calibrated probability/confidence score. | Not started |
| 25 | Example notebook with real outputs | End-to-end notebook on a real dataset (e.g. `sklearn.datasets.fetch_openml`), committed with executed outputs. | In progress |

Order: 19 (foundational — other backends benefit from generic model interface) → 25 (notebook, can run in parallel, no pipeline.py conflict) → 20/21/22 (scale, touch shared hot paths, do sequentially) → 23/24 (research-credibility, depend on stable model interface from 19).

## Execution model (once Wave -1 is merged)

- One feature = one branch (`feature/<short-name>`) + one worktree-isolated subagent + one PR against `main`.
- Each PR: implementation + config dataclass extension + pipeline wiring (default off) + tests + README update + ruff clean + (once Wave -1 lands) mypy clean + pytest green + coverage floor met.
- Run per-wave, not all-at-once — `config.py`/`pipeline.py`/`README.md` are touched by nearly every feature, so conflicts are expected and resolved PR-by-PR at merge, not by agents.
- **Do not start Wave 1 agents until Wave -1 items 1-5 are merged to main** — every feature PR after that point gets checked by the hardened CI gate.
