# Security policy

`samplepaths` generates random numbers for simulation. It opens no files or network connections and runs no code other than its own, NumPy and SciPy.

Its random numbers come from NumPy's generators, which are not cryptographically secure. Do not use realizations as keys, tokens or secrets; that they are predictable from the seed is intended and not a vulnerability.

## Supported versions

Fixes are released for the latest version on [PyPI](https://pypi.org/project/samplepaths/) only. Upgrade to it before reporting.

## Reporting a vulnerability

Do not open a public issue. Report privately with [Report a vulnerability](https://github.com/samplepaths/samplepaths/security/advisories/new) on the Security tab of the repository, with:

- the version of `samplepaths`, Python, NumPy and SciPy
- a short piece of code which shows the problem
- what an attacker could do with it

You can expect an answer within a week. When a report is confirmed, the fix is released as a new version, and an advisory which credits you, if you wish, is published with it.

A problem which is not a vulnerability, such as a wrong result or a crash on invalid parameters, is an ordinary bug: open an [issue](https://github.com/samplepaths/samplepaths/issues).
