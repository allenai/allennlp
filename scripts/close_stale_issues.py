from datetime import datetime as dt
import os

from github import Github


LABELS_TO_EXEMPT = ["contributions welcome", "merge when ready", "under development", "help wanted"]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("allenai/allennlp")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        if (
            issue.milestone is None
            and not issue.assignees
            and (dt.utcnow() - issue.updated_at).days > 7
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
            and issue.pull_request is None
        ):
            print("Closing", issue)
            #  issue.create_comment(
            #      "This issue is being closed due to lack of activity. "
            #      "If you think it still needs to be addressed, please comment on this thread."
            #  )
            #  issue.add_to_labels("stale")


if __name__ == "__main__":
    main()
