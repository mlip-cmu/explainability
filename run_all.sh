#!/usr/bin/env bash
# Run every demo and notebook, or only the folders given as arguments.
#   ./run_all.sh                      all folders
#   ./run_all.sh 04-lime-and-shap     one folder
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg

folders=("$@")
if [ ${#folders[@]} -eq 0 ]; then
    folders=(recidivism-data [0-9][0-9]-*/)
fi

for folder in "${folders[@]}"; do
    folder=${folder%/}
    echo "=== $folder ==="
    pushd "$folder" > /dev/null
    if [ "$folder" = recidivism-data ]; then
        uv run python -m recidivism_data
        uv run pytest -q
    else
        for script in *.py; do
            [ -e "$script" ] || continue
            [ "$script" = rule_list.py ] && continue   # a module, not a demo
            echo "--- $folder/$script"
            uv run "$script"
        done
        for notebook in *.ipynb; do
            [ -e "$notebook" ] || continue
            echo "--- $folder/$notebook"
            uv run jupyter nbconvert --to notebook --execute --inplace "$notebook"
        done
    fi
    popd > /dev/null
done
