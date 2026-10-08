# Contributing

Thanks for improving `meetingrec`.

## Development Setup

```bash
uv sync --group dev
uv run meetingrec doctor
```

Before opening a pull request, run the tests and lint:

```bash
uv run pytest -q
uv run ruff check src tests && uv run ruff format --check src tests
```

## Pull Request Expectations

- Do not commit recordings, transcripts, summaries, model weights, `.env` files, or
  credentials.
- Keep generated artifacts outside the repo or under ignored paths.
- Document new runtime dependencies in `README.md`.
- Update `AGENTS.md` when architecture, commands, or conventions change.

## Security

If you find a secret, private recording, transcript, or other sensitive material in
history, do not open a public issue with the details. Contact the maintainer privately.
