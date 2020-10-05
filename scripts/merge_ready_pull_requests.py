"""
Update and merge pull requests that are ready.
"""

import os

from github import Github


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("allenai/allennlp")
    open_prs = repo.get_pulls(state="open")

    for pr in open_prs:
        labels == [label.name.lower() for label in pr.get_labels()]
        # We only update or merge PRs that have the 'merge when ready' label.
        if "merge when ready" not in labels:
            continue

        if pr.mergeable_state == "behind":
            print("Updating", pr)
            pr.update_branch()
        elif pr.mergeable_state == "clean":
            print("Merging", pr)
            pr.merge()


if __name__ == "__main__":
    main()
