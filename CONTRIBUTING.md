# Contributing

*Thanks for being there!*

There are many ways you can help in the OpenNMT-tf project. This document will guide you through this process.

## Reporting issues

We use GitHub issues for bugs in the code that are **reproducible**. A good bug report should contain every information needed to reproduce it. Before opening a new issue, make sure to:

* **use the GitHub issue search** for existing and fixed bugs;
* **check if the issue has been fixed** in a more recent version;
* **isolate the problem** to give as much context as possible.

If you have questions on how to use the project or have trouble getting started with it, consider using [our forum](http://forum.opennmt.net/) instead and tagging your topic with the *opennmt-tf* tag.

## Requesting features

Do you think a feature is missing or would be a great addition to the project? Please open a GitHub issue to describe it.

## Developing code

*You want to share some code, that's great!*

* If you want to contribute with code but are unsure what to do,
  * search for *TODO* comments in the code: these are small dev tasks that should be addressed at some point
  * look for GitHub issues marked with the *help wanted* label: these are developments that we find particularly suited for community contributions.
* If you are planning to make a large change to the existing code, consider asking first on [the forum](http://forum.opennmt.net/) or [Gitter](https://gitter.im/OpenNMT/OpenNMT-tf) to confirm that it is welcome.

In any cases, your new code **must**:

* pass the existing tests

and **should**:

* pass the [`pylint`](https://www.pylint.org/) code checker (we use the TensorFlow Python coding style with relaxed line length)
* add new tests
* update the [documentation](docs/README.md)

## Testing

Tests are located in the `opennmt/tests` directory. Running them requires the following Python packages:

```bash
pip install parameterized nose2
```

* To run all tests:

```bash
nose2
```

* To run a single test file:

```bash
python -m opennmt.tests.model_test
```

* To run a single test case:

```bash
python -m opennmt.tests.model_test ModelTest.testCreateVariables
```

## Helping others

The project supports many model and training configurations. Sharing experiences (for example on the [forum](http://forum.opennmt.net/)) with existing or new configurations is highly appreciated.

Also, people often ask for help or suggestions on the [forum](http://forum.opennmt.net/) or on [Gitter](https://gitter.im/OpenNMT/OpenNMT-tf). Consider visiting them regularly and help some of them!
