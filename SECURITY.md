# Security Policy

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: [report a vulnerability](https://github.com/SeanPeppers/TreeShaker/security/advisories/new). Include reproduction steps and the affected version. Expect a first response within 7 days.

## Supported versions

Only the latest release on `main` receives security fixes.

## Dependency and code scanning

- `pip-audit` and `dependency-review-action` check dependencies on every pull request, and Dependabot opens weekly update PRs (`.github/dependabot.yml`).
- CodeQL, zizmor, gitleaks, and OpenSSF Scorecard run in CI (`.github/workflows/`).
- `main` requires a pull request and passing status checks before merge.
