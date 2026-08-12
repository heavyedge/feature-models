set -eu

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 SOURCE_STORAGE DESTINATION_PATH" >&2
  exit 2
fi

source_storage="$1"
destination_path="$2"
destination_dir="$(dirname "$destination_path")"
mkdir -p "$destination_dir"

temporary_path="$(mktemp "$destination_dir/.optuna.db.XXXXXX")"
cleanup() {
  rm -f "$temporary_path"
}
trap cleanup EXIT INT TERM

python3 - "$source_storage" "sqlite:///$temporary_path" <<'PY'
import sys

import optuna

source_storage, destination_storage = sys.argv[1:]
optuna.storages.RDBStorage(destination_storage)
studies = optuna.get_all_study_summaries(
    storage=source_storage,
    include_best_trial=False,
)
for study in studies:
    optuna.copy_study(
        from_study_name=study.study_name,
        from_storage=source_storage,
        to_storage=destination_storage,
    )
print(f"Exported {len(studies)} Optuna studies to {destination_storage}")
PY

mv "$temporary_path" "$destination_path"
trap - EXIT INT TERM
