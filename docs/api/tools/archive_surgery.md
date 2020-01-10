# allennlp.tools.archive_surgery

Helper script for modifying config.json files that are locked inside
model.tar.gz archives. This is useful if you need to rename things or
add or remove values, usually because of changes to the library.

This script will untar the archive to a temp directory, launch an editor
to modify the config.json, and then re-tar everything to a new archive.
If your $EDITOR environment variable is not set, you'll have to explicitly
specify which editor to use.

