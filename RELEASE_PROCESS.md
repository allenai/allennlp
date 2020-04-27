# AllenNLP GitHub and PyPI Release Process

This document describes the procedure for releasing new versions of the core library.
Most of the heavy lifting is actually done on GitHub Actions.
All you have to do is ensure the version in `allennlp/version.py` matches the target release version
and then trigger a GitHub release with the right tag.

> ❗️ This assumes you are using a clone of the main repo with the remote `origin` pointed
to `git@github.com:allenai/allennlp.git` (or the `HTTPS` equivalent).

The format of the tag should be `v{VERSION}`, i.e. the intended version of the release preceeded with a `v`.
So for the version `1.0.0` release the tag will be `v1.0.0`.

To make things easier, start by setting the tag to an environment variable, `TAG`.
Then you can copy and paste the commands below without worrying about mistyping the tag.

## Steps

1. Update `allennlp/version.py` (if needed) with the correct version and then commit and push the changes with:

    ```
    git commit -a -m "Update version for release $TAG"
    git push
    ```
    
    At this point `echo $TAG` should exactly match the output of `./scripts/get_version.py current`.

2. Then add the tag in git to mark the release:

    ```
    git tag $TAG -m "Release $TAG"
    ```

3. Push the tag to the main repo.

    ```
    git push --tags origin master
    ```

4. Find the tag you just pushed [on GitHub](https://github.com/allenai/allennlp/tags) and
click edit. Then add some release notes including the commit history since the last release which you can get with

    ```bash
    git log `git describe --always --tags --abbrev=0 HEAD^^`..HEAD^ --oneline
    ```

    Or, if you're using fish,

    ```fish
   git log (git describe --always --tags --abbrev=0 HEAD^^)..HEAD^ --oneline
   ```

5. Click "Publish Release", and if this is a pre-release make sure you check that box.

That's it! GitHub Actions will handle the rest.

## Fixing a failed release

If for some reason the GitHub Actions release workflow failed with an error that needs to be fixed, you'll have to delete both the tag and corresponding release from GitHub. After you've pushed a fix, delete the tag from your local clone with

```bash
git tag -l | xargs git tag -d && git fetch -t
```

Then repeat the steps above.
