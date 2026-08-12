set -eu

artifact_type="$1"

source_dir="${artifact_type}/${MAJOR_VERSION}"
if [ ! -d "$source_dir" ]; then
  echo "Skipping ${artifact_type}: ${source_dir} does not exist."
  exit 0
fi

asset_name="${artifact_type}-${GITHUB_RELEASE_TAG_NAME}.tar.gz"
archive_file="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$archive_file"' EXIT INT TERM

tar -C "$source_dir" -czf "$archive_file" .
source .github/scripts/app-token.sh

existing_asset_id="$(curl --fail --silent --show-error \
  --header "Authorization: Bearer $installation_token" \
  --header 'Accept: application/vnd.github+json' \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/${GITHUB_RELEASE_ID}/assets?per_page=100" | \
  python -c 'import json, sys; asset_name = sys.argv[1]; print(next((asset["id"] for asset in json.load(sys.stdin) if asset["name"] == asset_name), ""))' \
  "$asset_name")"

if [ -n "$existing_asset_id" ]; then
  curl --fail --silent --show-error --request DELETE \
    --header "Authorization: Bearer $installation_token" \
    --header 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/assets/${existing_asset_id}"
fi

encoded_asset_name="$(python -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$asset_name")"
curl --fail --silent --show-error --request POST \
  --header "Authorization: Bearer $installation_token" \
  --header 'Accept: application/vnd.github+json' \
  --header 'Content-Type: application/gzip' \
  --data-binary "@$archive_file" \
  "https://uploads.github.com/repos/${GITHUB_REPOSITORY}/releases/${GITHUB_RELEASE_ID}/assets?name=${encoded_asset_name}"

echo "Uploaded ${asset_name}."
