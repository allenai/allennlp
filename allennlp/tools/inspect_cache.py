import os

from allennlp.common.file_utils import CACHE_DIRECTORY
from allennlp.common.file_utils import filename_to_url


def main():
    print(f"Looking for datasets in {CACHE_DIRECTORY}...")
    if not os.path.exists(CACHE_DIRECTORY):
        print("Directory does not exist.")
        print("No cached datasets found.")

    cached_files = os.listdir(CACHE_DIRECTORY)

    if not cached_files:
        print("Directory is empty.")
        print("No cached datasets found.")

    for filename in cached_files:
        if not filename.endswith("json"):
            url, etag = filename_to_url(filename)
            print("Filename: %s" % filename)
            print("Url: %s" % url)
            print("ETag: %s" % etag)
            print()


if __name__ == "__main__":
    main()
