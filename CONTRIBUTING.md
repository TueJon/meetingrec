# Contributing

Thanks for improving `meetingrec`.

## Development Setup

```bash
uv sync
uv run meetingrec --help
```

Run a syntax check before opening a pull request:

```bash
uv run python -m compileall meetingrec.py
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
