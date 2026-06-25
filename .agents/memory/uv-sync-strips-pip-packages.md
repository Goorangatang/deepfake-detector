---
name: uv sync strips pip-installed packages during build
description: During production builds, uv sync removes packages not in pyproject.toml (including huggingface_hub installed via pip into .pythonlibs/). Build steps that need those packages must reinstall them via pip first.
---

# uv sync strips pip-installed packages during build

## The rule
Any production build step that depends on a package installed via plain pip (not declared in `pyproject.toml`) must explicitly reinstall it via pip at the start of that build step, because `uv sync` runs before build steps and removes undeclared packages.

**Why:** The build process runs `uv sync` early, which reconciles `.pythonlibs/` against `pyproject.toml`. Packages installed manually with `.pythonlibs/bin/python -m pip install` (like `huggingface_hub`) are not in `pyproject.toml`, so `uv sync` removes them. By the time `[services.production.build]` runs, those packages are gone.

**How to apply:** Prefix the build step command with a pip reinstall:
```
/home/runner/workspace/.pythonlibs/bin/python -m pip install --quiet <package> && <rest of build command>
```

This applies to the TrueSource AI artifact's model pre-download step. The pattern:
1. `uv sync` removes `huggingface_hub` (not in pyproject.toml)
2. Build step reinstalls it with pip
3. Build step downloads the model into HF cache
4. Everything gets baked into the image — runtime has both the package and the cached model
