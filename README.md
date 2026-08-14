# apb2

Convert proteomics software output to AnnData (rules-driven parser, second generation)

## Development

```bash
uv sync --group dev
make check
.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
```

All Python commands run from the synchronized project `.venv`.
