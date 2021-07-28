# AllenNLP GitHub and PyPI Release Process

This document describes the procedure for releasing new versions of the core library.

> ❗️ This assumes you are using a clone of the main repo with the remote `origin` pointed
to `git@github.com:allenai/allennlp.git` (or the `HTTPS` equivalent).

## Steps

1. Set the environment variable `TAG`, which should be of the form `v{VERSION}`.

    For example, if the version of the release is `1.0.0`, you should set `TAG` to `v1.0.0`:

    ```bash
    export TAG='v1.0.0'
    ```

    Or if you use `fish`:

    ```fish
    set -x TAG 'v1.0.0'
    ```

2. Update `allennlp/version.py` with the correct version. Then check that the output of

    ```
    python scripts/get_version.py current
    ```

    matches the `TAG` environment variable.

3. Update the `CHANGELOG.md` so that everything under the "Unreleased" section is now under a section corresponding to this release.

4. Update the `CITATION.cff` file to refer to the right version.

4. Commit and push these changes with:

    ```
    git commit -a -m "Prepare for release $TAG" && git push
    ```
    
5. Then add the tag in git to mark the release:

    ```
    git tag $TAG -m "Release $TAG" && git push --tags
    ```

6. Find the tag you just pushed [on GitHub](https://github.com/allenai/allennlp/tags), click edit, then copy over the output of:

    ```
    python scripts/release_notes.py
    ```

    On a Mac, for example, you can just pipe the above command into `pbcopy`.

7. Check the box "This is a pre-release" if the release is a release candidate (ending with `rc*`). Otherwise leave it unchecked.

8. Click "Publish Release". GitHub Actions will then handle the rest, including publishing the package to PyPI the Docker image to Docker Hub.


9. After the [GitHub Actions workflow](https://github.com/allenai/allennlp/actions?query=workflow%3AMaster+event%3Arelease) finishes, follow the same process to publish a release for the `allennlp-models` repo.


## Fixing a failed release

If for some reason the GitHub Actions release workflow failed with an error that needs to be fixed, you'll have to delete both the tag and corresponding release from GitHub. After you've pushed a fix, delete the tag from your local clone with

```bash
git tag -l | xargs git tag -d && git fetch -t
```

Then repeat the steps above.
