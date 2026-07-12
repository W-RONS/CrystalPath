#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
APP_PATH="$ROOT_DIR/dist/macos/CrystalPath.app"
RELEASE_DIR="$ROOT_DIR/release"
ARCHIVE_PATH="$RELEASE_DIR/CrystalPath-0.3.5-macos-arm64.zip"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/private/tmp}/crystalpath-release.XXXXXX")"
STAGED_APP="$STAGING_DIR/CrystalPath.app"

cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

if [[ ! -d "$APP_PATH" ]]; then
  echo "Build the app first: ./packaging/macos/build_app.sh" >&2
  exit 1
fi

mkdir -p "$RELEASE_DIR"
ditto --noextattr --norsrc "$APP_PATH" "$STAGED_APP"
xattr -cr "$STAGED_APP"
codesign --force --deep --sign - "$STAGED_APP"
codesign --verify --deep --strict "$STAGED_APP"
ditto -c -k --sequesterRsrc --keepParent "$STAGED_APP" "$ARCHIVE_PATH"
(
  cd "$RELEASE_DIR"
  shasum -a 256 "$(basename "$ARCHIVE_PATH")" > "$(basename "$ARCHIVE_PATH").sha256"
)

echo "Created $ARCHIVE_PATH"
