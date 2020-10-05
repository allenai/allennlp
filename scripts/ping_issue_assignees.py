"""
Ping issue assignees for non-milestone issues that have had no activity in the past 14 days.
"""

from datetime import datetime as dt
import os

from github import Github


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("allenai/allennlp")
    open_issues = repo.get_issues(
        state="open", milestone="none", assignee="*", sort="updated", direction="asc"
    )

    for issue in open_issues:
        # We only look at issues that have been inactive for the past 14 days.
        # And since these will be sorted by last updated time, it's safe to break
        # out of this loop as soon as we encounter the first issue that's been updated
        # within the past 14 days.
        if (dt.utcnow() - issue.updated_at).days < 14:
            break
        # Skip issue if it's actually a pull request.
        if issue.pull_request is not None:
            continue
        assignees = ", ".join([f"@{user.login}" for user in issue.assignees])
        print(f"Pinging {assignees} for {issue}")
        issue.create_comment(
            f"{assignees} this is just a friendly ping to make sure you "
            "haven't forgotten about this issue 😜"
        )


if __name__ == "__main__":
    main()
