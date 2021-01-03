#!/usr/bin/env bash

# if any command inside script returns error, exit and return that error 
set -e

# magic line to ensure that we're always inside the root of our application,
# no matter from which directory we'll run script
# thanks to it we can just enter `./scripts/run-tests.bash`
cd "${0%/*}/.."

# let's fake failing test for now 
echo "Running tests"
echo "............................" 

SIZE=$1

# Get the current branch.
# branch=$(git branch | grep \* | cut -d ' ' -f2)
declare -a large_files=()
# Get all changed files (compared to main branch) 
for path in $(git diff --name-only main | sed -e 's/A[[:space:]]//'); 
do
     # Check to see if any sizes are greater than 2MB
    large_files+=($(du -m $path | awk -v size="$SIZE" '{if ($1 > size) print $2}'))
done

# Result
if [ ${#large_files[@]} -gt 0 ];
then
    # Display result
    echo "Found ${#large_files[@]} files have size bigger than "$SIZE" MB" 
    echo "--------------------------------------------------------"
    for file in ${large_files[@]};
    do 
        echo $file
    done
    echo "--------------------------------------------------------"
    echo "Please reduce file size before commit."
    echo "Failed to commit!" && exit 1
else
    echo "Passed" && exit 0
fi
