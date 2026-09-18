# Contributing to hexmb

Contributions, bug reports, and questions are welcome.

## Report a bug or ask a question

Open an issue on GitHub: <https://github.com/JieWangnk/hexmb/issues>. For a bug,
please include the command or short script you ran, the full error, and your
Python and hexmb versions. A minimal example that reproduces the problem is the
fastest way to a fix.

## Contribute code

1. Fork the repository and create a branch for your change.
2. Install the dev environment: `pip install -e ".[scipy,dev]"`.
3. Add or update tests under `tests/` for your change, and run `python -m pytest tests/ -q`.
4. Keep the style of the surrounding code (numpy-only core; `black`/`ruff`, line length 100).
5. Open a pull request describing what changed and why.

New geometry templates are the most useful additions: a template consumes a
surface or a centreline and returns an assembled `MultiBlockMesh` (see
`hexmb/templates/tube.py` and `hexmb/templates/lv.py`).

## Seek support

If something is unclear in the README or you are unsure whether hexmb fits your
geometry, open an issue with the "question" label; the applicability domain is
described in the README under *Conditions of use*.
