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
| 1 | **Tests in CI** | Add `pytest` step to `.github/workflows/lint.yml` (or new `test.yml`) — currently only `ruff check`/`ruff format --check` run. Broken code currently ships if lint-clean. | High |
| 2 | **Type checking** | Add `mypy` (or `pyright`) as dev dep + CI step. ML libs (numpy/pandas/xgboost) drift dtypes/shapes silently — catches that class of bug pre-merge. | High |
| 3 | **Dependency pinning** | Lock exact versions — `uv lock` (preferred, fast) or generate `requirements-lock.txt` via `pip freeze` in a clean venv. `pyproject.toml` deps are all loose `>=` ranges — CI today can silently pick up a breaking upstream release. | High |
| 4 | **Coverage threshold** | `pytest-cov` already installed (`pyproject.toml` line 35) but no floor enforced. Add `--cov-fail-under=80` (or measure current baseline first, set realistic floor) to `addopts`. CI fails if coverage drops. | Medium |
| 5 | **Security scan** | Add `pip-audit` step to CI — one-liner, checks deps against known CVE advisories. | Medium |
| 6 | **Branch protection** | GitHub → Settings → Branches → `main` → require all CI status checks (ruff, pytest, mypy, pip-audit) green before merge. Manual GitHub setting, not code — do this once infra checks above exist and are green. | Lower (manual, last step of this wave) |

Order: 1 → 2 → 3 → 4 → 5 as separate PRs (small, fast to review, avoids one giant CI-rewrite PR). 6 done manually after 1-5 merged.

---

## Wave 1 — Detection/scoring additions (low conflict risk, extend existing dataclass pattern)

| # | Feature | Notes |
|---|---|---|
| 1 | VIF multicollinearity detection | Extends leakage checks beyond pairwise corr — catches multi-feature redundancy clusters |
| 2 | SHAP interaction values | New output: pairwise feature interaction strengths |
| 3 | Mutual information scoring | Catches non-tree-friendly / categorical dependency signal SHAP+perm miss |
| 4 | Bootstrap ensemble importance | Mean/std of SHAP+perm importance across resampled fits — quantifies ranking stability |
| 5 | Class imbalance detection/warning | Flags extreme class ratios that break permutation-importance assumptions |
| 6 | Feature groups / hierarchical tags | Tag features (e.g. `price_*`) to aggregate importance at group level |

## Wave 2 — Alternative algorithms (behind flags, bigger pipeline touch)

| # | Feature | Notes |
|---|---|---|
| 7 | BorutaSHAP `--use-boruta` flag | Pending decision from earlier — adds `boruta_decision` column (Confirmed/Tentative/Rejected) |
| 8 | RFECV option | Iterative elimination backed by CV score |
| 9 | LightGBM backend | Optional model swap to cross-validate importance rankings, faster fit |
| 10 | CatBoost backend | Native categorical handling, skip one-hot |
| 11 | Random forest baseline importance | Cross-check tree-model ranking against RF |
| 12 | Model-agnostic sklearn permutation importance | Support any sklearn estimator, not just XGBoost |

## Wave 3 — Performance/scale

| # | Feature | Notes |
|---|---|---|
| 13 | GPU acceleration flag | `tree_method=gpu_hist` / RAPIDS path |
| 14 | Optuna pruning + early stopping | Skip plateauing trials, cut tuning time |
| 15 | Large-dataset sampling strategy | Auto-stratified sampling above a row-count threshold |

## Wave 4 — Output/UX & specialized checks

| # | Feature | Notes |
|---|---|---|
| 16 | Interactive HTML dashboard (plotly) | Sortable importances, color-coded leakage/recommendation |
| 17 | Time-series leakage checks | Look-ahead leakage, temporal autocorrelation dominance |
| 18 | MLflow/DVC export | Log rankings + run metadata for experiment tracking |

---

## Execution model (once Wave -1 is merged)

- One feature = one branch (`feature/<short-name>`) + one worktree-isolated subagent + one PR against `main`.
- Each PR: implementation + config dataclass extension + pipeline wiring (default off) + tests + README update + ruff clean + (once Wave -1 lands) mypy clean + pytest green + coverage floor met.
- Run per-wave, not all-at-once — `config.py`/`pipeline.py`/`README.md` are touched by nearly every feature, so conflicts are expected and resolved PR-by-PR at merge, not by agents.
- **Do not start Wave 1 agents until Wave -1 items 1-5 are merged to main** — every feature PR after that point gets checked by the hardened CI gate.
