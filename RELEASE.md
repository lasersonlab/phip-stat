# Release protocol

1.  Ensure that the help output in the README file is up-to-date with any
    changes.

2.  Generate a summary of all the commits since the last release

    ```bash
    git log $LAST_RELEASE_TAG..HEAD
    ```

3.  Set the release version in `setup.py` (remove the `.dev0` tag if
    applicable) and commit the version number change.

4.  Tag version number and summarize changes in the tag message

    ```bash
    git tag -a vX.Y.Z
    ```

5.  Push the tag upstream

    ```bash
    git push upstream vX.Y.Z
    ```

    or

    ```bash
    git push upstream --tags
    ```

6.  Create the distributions

    ```bash
    python setup.py sdist bdist_wheel
    ```

8.  Upload the distributions

    ```bash
    twine upload dist/*
    ```

7.  If working on master, bump up to the next anticipated version with a
    `.dev0` tag and commit


*Backporting*

1.  Checkout the tag for the version to backport onto and create a new branch

    ```bash
    git checkout vX.Y.Z
    git checkout -b backport
    ```

2.  Cherry pick the relevant commits onto the `backport` branch

3.  Goto #1 for main release flow

4.  Remove the `backport` branch
