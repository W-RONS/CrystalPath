#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${CRYSTALPATH_BUILD_PYTHON:-$(command -v python3)}"
BUILD_DIR="$ROOT_DIR/build/macos"
DIST_DIR="$ROOT_DIR/dist/macos"
ENTRYPOINT="$ROOT_DIR/packaging/macos/entrypoint.py"
APP_PATH="$DIST_DIR/CrystalPath.app"
PYINSTALLER_BIN="${CRYSTALPATH_PYINSTALLER_BIN:-$(dirname "$PYTHON_BIN")/pyinstaller}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$PYINSTALLER_BIN" ]]; then
  echo "PyInstaller is not installed next to $PYTHON_BIN" >&2
  echo "Install the project with the deploy extra first: python -m pip install -e '.[deploy]'" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR" "$DIST_DIR"
mkdir -p \
  "$BUILD_DIR/pyinstaller-config" \
  "$BUILD_DIR/matplotlib" \
  "$BUILD_DIR/pyvista-userdata"
export PYINSTALLER_CONFIG_DIR="$BUILD_DIR/pyinstaller-config"
export MPLCONFIGDIR="$BUILD_DIR/matplotlib"
export PYVISTA_USERDATA_PATH="$BUILD_DIR/pyvista-userdata"

if [[ -e "$APP_PATH" ]]; then
  echo "Output already exists: $APP_PATH" >&2
  echo "Move it aside before rebuilding so stale files cannot enter a release." >&2
  exit 1
fi

cd "$ROOT_DIR"
"$PYINSTALLER_BIN" \
  --noconfirm \
  --clean \
  --windowed \
  --onedir \
  --name CrystalPath \
  --distpath "$DIST_DIR" \
  --workpath "$BUILD_DIR/pyinstaller" \
  --specpath "$BUILD_DIR/pyinstaller-spec" \
  --paths "$ROOT_DIR/src" \
  --osx-bundle-identifier com.wrons.crystalpath \
  --target-arch arm64 \
  --collect-all vtkmodules \
  --collect-all pyvista \
  --collect-all pyvistaqt \
  --collect-all pymatgen \
  --hidden-import vtkmodules.qt.QVTKRenderWindowInteractor \
  --exclude-module pytest \
  --exclude-module pyvista.conftest \
  --exclude-module pyvista.ext \
  "$ENTRYPOINT"

if [[ ! -d "$APP_PATH" ]]; then
  echo "Expected application bundle was not created: $APP_PATH" >&2
  exit 1
fi

/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString 0.3.5" "$APP_PATH/Contents/Info.plist" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string 0.3.5" "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion 0.3.5" "$APP_PATH/Contents/Info.plist" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string 0.3.5" "$APP_PATH/Contents/Info.plist"

echo "Built $APP_PATH"
echo "Run ./packaging/macos/package_release.sh to create and verify an ad-hoc signed release archive."
