# Contributing to samplepaths

Bug reports, feature requests and pull requests are welcome, in particular for additional processes, generalizations and algorithms.

## Reporting a problem

Open an issue at https://github.com/samplepaths/samplepaths/issues with:

- the version of `samplepaths`, Python, NumPy and SciPy
- a short piece of code which shows the problem, with a seeded generator when it involves random results
- what you expected and what happened instead

Report a vulnerability privately instead, as [SECURITY.md](SECURITY.md) describes.

## Making a change

```shell
git clone https://github.com/samplepaths/samplepaths.git
cd samplepaths
make install  # create the environment with uv
make check    # lint, type check, test and build the docs
```

Run `make` to list all targets. A pull request should:

- keep `make check` passing
- add tests for the change; a fix comes with a test which fails without it
- leave the results of `sample()` and `sample_at()` for a given seed unchanged, or say in the changelog why they change
- update the documentation in the same pull request: the guide page of the feature, an example in `docs/examples` when it helps, and an "Unreleased" section in `docs/project/changelog.rst`
- put every new class in its own module

The [development page](https://samplepaths.readthedocs.io/en/latest/project/development.html) describes the layout of the project, the code style and how to add a process.

## License

By contributing you agree that your contribution is released under the MIT license of the project.
