from datetime import datetime as dt
import os

from github import Github


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("allenai/allennlp")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        if (
            issue.milestone is None
            and issue.assignees
            and issue.pull_request is None
            and (dt.utcnow() - issue.updated_at).days >= 14
        ):
            assignees = ", ".join([f"@{user.login}" for user in issue.assignees])
            print(f"Pinging {assignees} for {issue}")
            issue.create_comment(
                f"{assignees} this is just a friendly ping to make sure you "
                "haven't forgotten about this issue 😜"
            )


if __name__ == "__main__":
    main()
