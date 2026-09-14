#!/bin/bash

cd "$(dirname "$0")"

if [ -z "$1" ]; then
    echo "Error: Please specify a file path to run."
    echo "Usage: ./run.sh src/bankhesit/test_bankheist.py"
    echo "   or: ./run.sh bankhesit/test_bankheist"
    exit 1
fi

input="$1"

# Strip leading ./ and src/ if provided by the user
input="${input#./}"
input="${input#src/}"

# Strip trailing .py extension if included
input="${input%.py}"

# Replace forward slashes with dots and prefix with Src.
module_path="src.${input//\//.}"

python -m "$module_path"