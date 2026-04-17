# Git Conventions for Amenity Detector

A reference for how we write commits, name branches, and manage PRs in this project.

---

## Commit message format

```
<type>: <short summary in present tense, under 72 chars>

<optional body — explain WHY, not what. Wrap at 72 chars.>
```

### Types

| Type       | When to use                                                  |
|------------|--------------------------------------------------------------|
| `feat`     | A new feature or capability                                  |
| `fix`      | A bug fix                                                    |
| `refactor` | Code restructuring with no behaviour change                  |
| `chore`    | Dependency updates, config changes, tooling                  |
| `test`     | Adding or updating tests                                     |
| `docs`     | Documentation only changes                                   |
| `ci`       | CI/CD pipeline changes (GitHub Actions, Docker, Makefiles)   |

### Rules
- Use **present tense**: "Add endpoint" not "Added endpoint"
- No full stop at the end of the subject line
- Keep the subject **under 72 characters**
- No bullet points in the subject
- If you need to explain more, add a blank line then a body paragraph

### Examples

```
feat: add Gemini 2.0 Flash VLM client with retry logic

The Gemini free tier caps at 1500 req/day so we add a hard retry
with exponential backoff to avoid silent failures at the boundary.
```

```
fix: set PYTHONPATH in Dockerfile so local modules resolve correctly
```

```
chore: migrate dependency management from pip to uv
```

```
refactor: replace Flask API layer with FastAPI and split into routers
```

---

## Branch naming

```
<type>/<short-description-in-kebab-case>
```

Examples:
- `feat/gradio-ui`
- `fix/ollama-timeout`
- `refactor/core-pipeline`
- `chore/upgrade-sqlalchemy`

---

## Workflow

```bash
# 1. Create a branch from main
git checkout main && git pull
git checkout -b feat/your-feature

# 2. Stage logically — commit one concern at a time
git add <specific files>
git commit -m "feat: describe the change"

# 3. Push and open a PR
git push -u origin feat/your-feature
gh pr create --title "feat: your feature" --body "..."

# 4. After PR is merged, clean up
git checkout main && git pull
git branch -d feat/your-feature
```

---

## What belongs in a single commit

- One concern per commit — don't mix a bug fix with a refactor
- Keep DB schema changes separate from API changes
- Keep dependency bumps (pyproject.toml / uv.lock) in a `chore` commit
- Tests that cover a feature can go in the same commit as the feature,
  or in a separate `test:` commit — be consistent

---

## What NOT to commit

- `.env` files or any secrets
- `*.db` / `*.db-journal` files (local SQLite dev files)
- `storage/` directory (runtime image uploads)
- `.venv/` or `__pycache__/`
