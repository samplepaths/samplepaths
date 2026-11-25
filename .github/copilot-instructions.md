# Instructions for coding agents

`samplepaths` generates realizations of stochastic processes as NumPy arrays. It is pure Python on NumPy and SciPy, and it is meant to stay that way.

## Commands

Everything runs through `uv` and the top-level `Makefile`. Do not add another task runner or per-folder make files.

```shell
make install   # create the environment from uv.lock
make check     # lint, type check, test and build the docs: run it before you finish
make format    # format and apply the safe lint fixes
make           # list all targets
```

The workflows in `.github/workflows` call make targets only. A step which needs more than one command is a Python script in `.github/scripts` with a make target of its own, not shell in the workflow.

## Rules for the code

- **American English** in code, comments, documentation and commit messages: color, behavior, modeling, analyze.
- **One class per file.** Every class, internal base classes included, has its own module named after it in snake case. Modules of helper functions without a class are fine and start with an underscore, such as `_recurrence.py`.
- **Files stay under 300 lines.** Before a file reaches that size, move helpers to a private module or split it by topic. Several older files are still over the limit; split one when you change it.
- **Standard tooling only.** `ruff` for lint and format, `mypy` in strict mode, `pytest`. No unusual configuration.
- Every public module, class, method and function has a docstring with Sphinx field lists (`:param name:`, `:returns:`, `:raises Error:`). Types are in the hints, not in the docstrings.
- Constructors take keyword-only parameters, and parameters are validated in property setters with the functions of `samplepaths.utils.validation`.
- Import `override` from `samplepaths._compat`, which supports Python 3.11.

## Seeded results must not change

`sample()` and `sample_at()` return the same values, bit for bit, for the same seed from one version to the next. This is a promise to users.

- An optimization must compute the same values in the same order of floating point operations. Check it against the previous version on seeded realizations before and after the change.
- A better algorithm goes into a new method, not into `sample()`. This is why exact sampling is `sample_exact()` and `sample_paths_exact()`.
- A fix which has to change results is allowed. The changelog then says which process, what was wrong and how results change.
- `sample_paths()` may draw its random numbers in a different order than repeated calls to `sample()`. It must give the same distribution, and the same result for the same seed and number of workers.

## Adding a process

1. Add a module with the class to the subpackage of its family: `continuous`, `diffusion`, `discrete`, `timeseries`, `multivariate` or `noise`. Derive from `BaseTimeProcess` for a process on a time interval or `BaseSequenceProcess` for one which evolves in steps.
2. Export it from the subpackage and from `samplepaths/__init__.py`.
3. Add a test module in the same layout under `tests`, with invalid parameters, the string representations, shapes, seeded values and a check against a known moment or distribution.
4. Add the process to `CASES` in `tests/test_sample_paths.py`, which compares `sample_paths()` with `sample()`.
5. Add an example named `docs/examples/<family>/plot_<module>.py`; a test fails without it.
6. Override `_sample_paths` when all realizations can be generated together. A loop over steps in Python with all realizations in one array is the usual way. Check on a large batch whether threads help before setting `_parallel_paths`: many short NumPy calls on small arrays get slower with more threads.

A process with several components returns one array with a component axis: `(components, n + 1)` from `sample()` and `(paths, components, n + 1)` from `sample_paths()`.

## Documentation is part of the change

- Update every affected page in the same change: the guide page in `docs/guide`, the performance and comparison pages when a number moves, the README and the "Unreleased" section of `docs/project/changelog.rst`.
- Rewrite and reorganize so that the docs read as one coherent guide. Do not append a section per change.
- Quote measured numbers only. Sampling times come from `docs/scripts/benchmark.py`, and the comparison with `stochastic-rs` from `docs/scripts/compare_rust.py` and `compare_rust_outputs.py`.
- The docs build with warnings as errors and run every example of the gallery.

## Git and releases

- `master` is protected: changes go through a pull request, merged by squashing. The maintainer merges; do not merge yourself.
- Commit, push or open a pull request only when asked to.
- The version is not written in any file. It is read from the git tag: `v0.6.0` gives `0.6.0`.
- To release: rename "Unreleased" in the changelog to the version and its date in a pull request; after it is merged, create a GitHub release on `master` with the tag `vX.Y.Z`. Publishing the release uploads to PyPI.

## Comparing with stochastic-rs

`stochastic-rs` is a package with a Rust core which serves as the reference for speed and for checking results. It is not a dependency.

- Its `n` is the number of points of a realization; here `n` is the number of increments.
- The two use different random number generators, so only distributions can be compared. A single small p-value among many comparisons is expected by chance; repeat with more realizations before concluding that two results differ.
- Parameters are often defined differently. The comparison script documents the mapping for each process.
