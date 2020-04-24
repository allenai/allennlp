# AllenNLP GitHub and PyPI Release Process

This document describes the procedure for releasing new versions of the core library.
Most of the heavy lifting is actually done on GitHub Actions.
All you have to do is trigger a GitHub release with the right tag.

> ❗️ This assumes you are using a clone of the main repo with the remote `origin` pointed
to `git@github.com:allenai/allennlp.git` (or the `HTTPS` equivalent).

The format of the tag should be `v{VERSION}`, i.e. the intended version of the release preceeded with a `v`.
So for the version `1.0.0` release the tag will be `v1.0.0`.

> ❗️ If this is a pre-release, such as `rc1`, it's not necessary to update `version.py`, as the pre-release suffix will be automatically taken from the tag. However, for this to work properly and to be consistent with our versioning conventions, make sure the suffix is separated with another `.`, for example
> - ✅ `v1.0.0.rc1` = good,
> - ❌ `v1.0.0rc1` = bad.
>
> Note that the release on PyPI may not have the dot before the suffix since PyPI has its own weird versioning rules.

To make things easier, start by setting the tag to an environment variable, `TAG`.
Then you can copy and paste the commands below without worrying about mistyping the tag.

## Steps

1.  Add the tag in git to mark the release:

    ```
    git tag $TAG -m "Release $TAG"
    ```

2. Push the tag to the main repo.

    ```
    git push --tags origin master
    ```

3. Find the tag you just pushed [on GitHub](https://github.com/allenai/allennlp/tags) and
click edit. Then add some release notes including the commit history since the last release which you can get with

    ```bash
    git log `git describe --always --tags --abbrev=0 HEAD^^`..HEAD^ --oneline
    ```

    Or, if you're using fish,

    ```fish
   git log (git describe --always --tags --abbrev=0 HEAD^^)..HEAD^ --oneline
   ```

4. Click "Publish Release", and if this is a pre-release make sure you check that box.

That's it! GitHub Actions will handle the rest.
