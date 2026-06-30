# Pre-public checklist

Things that are blocked or degraded right now because this repo is **private** on a free GitHub plan (no GitHub Advanced Security). Either upgrade to GitHub Pro/Team to unlock these on a private repo, or flip them on when this repo goes public — both unlock the same features for free.

## Blocked entirely (need GHAS or public)

- **Branch protection on `main`** (require PR review, require CI to pass before merge, block force-push). `gh api repos/.../branches/main/protection` 403s with "Upgrade to GitHub Pro or make this repository public."
- **`dependency-review-action` in CI** — was added then removed from `.github/workflows/lint.yml` because it hard-fails with "Dependency review is not supported on this repository" without the dependency graph enabled.
- **Private vulnerability reporting** (`/security/advisories/new`) — 404s, needs the dependency graph as a prerequisite.
- **Dependency graph itself** (`/dependency-graph/sbom` 404s) — the underlying feature both of the above need.

## Degraded (partially working)

- **zizmor** (workflow security scanner) — runs and fails CI on findings, but `advanced-security: "false"` is set so its SARIF results don't show up in the repo's Security tab, just in CI logs.
- **GitHub Pages / Read the Docs** — both free tiers require a public repo, so `docs/source/*.md` aren't hosted anywhere; README points at the source files directly with build-locally instructions instead.

## Do when going public

1. Re-add `dependency-review-action` job to `.github/workflows/lint.yml` (removed in the `ci/security-hardening` PR — check git history for the exact job config).
2. Set `advanced-security: "true"` (or drop the input) on the zizmor step so findings show in the Security tab.
3. Enable branch protection on `main`: require the `Lint` workflow's jobs to pass, require 1 approval, no force-push.
4. Enable dependency graph + Dependabot alerts + private vulnerability reporting in Settings → Security.
5. Point `SECURITY.md`'s reporting instructions back at `/security/advisories/new` instead of email.
6. Either register the project on Read the Docs, or set up a GitHub Pages deploy workflow for `docs/source/`, and restore the hosted-docs links in `README.md`.
7. Consider a `CODECOV_TOKEN` repo secret isn't actually needed for public repos on Codecov's free tier (tokenless uploads work) — can simplify the codecov step then.

## Not actually blocked by privacy (do anytime)

- **GitHub Sponsors button** — works on private repos too, just needs `.github/FUNDING.yml` with a `github: [username]` entry, plus the user/org being accepted into the GitHub Sponsors program separately (that's an application, not a paid tier).

## Do before public release (not privacy-related, just not ready yet)

- **PyPI package publish** — no `__version__` string, no release workflow, no changelog. Currently the only way to install is `pip install -e .` from a git checkout. Before announcing publicly: add a `release.yml` workflow (build sdist+wheel, `pypa/gh-action-pypi-publish` with trusted publishing — no API token needed), tag-triggered, and bump `version` in `pyproject.toml` per release.
- **Hosted docs** — covered above (GitHub Pages / RTD), but also blocked on the repo not being ready for public eyes yet, not just the privacy/GHAS issue.
- **CHANGELOG.md** — none exists yet; needed once there's a versioned release history to track.
