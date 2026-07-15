# horus-docker

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A [horus-runtime](https://github.com/temple-compute/horus-runtime) plugin that runs tasks inside Docker containers.

---

## Overview

**horus-docker** contributes a `DockerExecutor` to the `horus.executor` entry point group. Once installed, any `HorusContext` that calls `HorusContext.boot()` will automatically discover and register the executor — no manual wiring required.

The executor accepts a `CommandRuntime`-produced shell command and runs it via `/bin/sh -c` inside a fresh Docker container, giving you the same shell semantics (pipes, globbing, `&&`, …) as the local shell executor.

---

## Repository structure

```
src/
└── horus_docker/
    ├── __init__.py
    ├── i18n.py                  # plugin-scoped gettext wrapper
    ├── locale/
    │   └── messages.pot         # translatable strings template
    └── executor/
        ├── __init__.py
        └── docker.py            # DockerExecutor implementation
tests/
├── __init__.py
├── conftest.py                  # shared fixtures (registry, HorusContext)
└── unit/
    ├── __init__.py
    └── test_docker_executor.py
babel.cfg
Makefile
pyproject.toml
```

---

## DockerExecutor

Registered under `kind = "docker"`. Only accepted when the task's runtime is a `CommandRuntime`.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | `str` | required | Docker image to run (e.g. `python:3.13-slim`) |
| `env` | `dict[str, str]` | `{}` | Environment variables passed into the container |
| `volumes` | `dict[str, str]` | `{}` | Bind mounts as `host_path → container_path` (read-write) |
| `ports` | `dict[str, str]` | `{}` | Port mappings as `host_port → container_port` |
| `working_dir` | `str \| None` | `None` | Working directory inside the container |
| `network` | `str \| None` | `None` | Docker network to attach the container to |
| `entrypoint` | `str \| list[str] \| None` | `None` | Override the image's `ENTRYPOINT` |
| `user` | `str \| None` | `None` | User (`name`/`uid`, optionally `uid:gid`) to run as |
| `auto_remove` | `bool` | `True` | Remove the container after execution finishes |

### Behaviour

1. The task's `CommandRuntime` prepares the shell command via `setup_runtime()`.
2. The command is wrapped as `["/bin/sh", "-c", <command>]` and passed to `docker-py`'s `containers.run()`.
3. The container runs to completion (blocking, dispatched to a thread with `asyncio.to_thread`).
4. Container logs are captured and emitted at `DEBUG` level.
5. If the Docker daemon raises a `DockerException`, or the container exits with a non-zero status code, a `TaskExecutionError` is raised.

---

## Development

### Requirements

- Python ≥ 3.13
- Docker daemon accessible from the host
- `horus-runtime` ≥ 0.1.1 (install from source or PyPI)

### Setup

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Common commands

| Command | Description |
|---|---|
| `make test` | Run the full test suite with coverage |
| `make lint` | ruff + mypy |
| `make format` | Auto-fix with ruff |
| `make type-check` | mypy only |
| `make babel-extract` | Update `messages.pot` |
| `make babel-add LANG=es` | Add a new language |
| `make babel-check` | Verify all strings are translated |
| `make clean` | Remove build artefacts and caches |

---

## Internationalization (i18n)

Each plugin maintains its **own** gettext domain and locale directory, independent of the runtime's translations.

`src/horus_docker/i18n.py` wraps Python's `gettext` module, looking for compiled `.mo` files in `src/horus_docker/locale/<lang>/LC_MESSAGES/horus_docker.mo`. If no catalog exists for the detected locale, it falls back to the original string.

Import the wrapper as `_` (required by Babel's extractor) in any module with user-visible strings. Use `make babel-extract` → edit `.po` → `make babel-check` to update translations. The pre-commit hook prevents committing incomplete catalogs.

> Full i18n workflow and plural-form reference: [docs.templecompute.com](https://docs.templecompute.com/docs/sdk/i18n).

---

## License

MIT — see [LICENSE](LICENSE).
