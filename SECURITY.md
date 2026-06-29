# Security Policy

## Reporting a vulnerability

Open a private [GitHub security advisory](https://github.com/SeanPeppers/TreeShaker/security/advisories/new) rather than a public issue. Include reproduction steps and affected version.

## Dependency scanning

`pip-audit` runs in CI on every push and pull request against `main` (see `.github/workflows/lint.yml`), and Dependabot opens weekly PRs for outdated dependencies (`.github/dependabot.yml`).
