"""
Close stale issues.
"""

from datetime import datetime as dt
import os

from github import Github


LABELS_TO_EXEMPT = ["contributions welcome", "merge when ready", "under development", "help wanted"]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("allenai/allennlp")
    open_issues = repo.get_issues(
        state="open", milestone="none", assignee="none", sort="updated", direction="asc"
    )

    for issue in open_issues:
        # We only look at issues that have been inactive for the past 7 days.
        # And since these will be sorted by last updated time, it's safe to break
        # out of this loop as soon as we encounter the first issue that's been updated
        # within the past 7 days.
        if (dt.utcnow() - issue.updated_at).days <= 7:
            break
        # Skip issue if it's actually a pull request.
        if issue.pull_request is not None:
            continue
        # Skip issue if it has one of the special labels.
        if any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels()):
            continue
        # Skip issue if it was created within the past 14 days.
        if (dt.utcnow() - issue.created_at).days < 14:
            continue
        print("Closing", issue)
        issue.create_comment(
            "This issue is being closed due to lack of activity. "
            "If you think it still needs to be addressed, please comment on this thread 👇"
        )
        issue.add_to_labels("stale")
        issue.edit(state="closed")


if __name__ == "__main__":
    main()
