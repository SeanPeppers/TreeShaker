# Security Policy

## Reporting a vulnerability

GitHub's private security advisory flow requires the dependency graph, which isn't available on a private repo without GitHub Advanced Security. Until this repo goes public (see [PRE_PUBLIC_CHECKLIST.md](PRE_PUBLIC_CHECKLIST.md)), email **gagepeppers@gmail.com** instead of filing a public issue. Include reproduction steps and affected version.

## Dependency scanning

`pip-audit` runs in CI on every push and pull request against `main` (see `.github/workflows/lint.yml`), and Dependabot opens weekly PRs for outdated dependencies (`.github/dependabot.yml`).
