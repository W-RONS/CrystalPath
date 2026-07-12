#!/bin/zsh
set -e
cd "$(dirname "$0")"
exec "${CRYSTALPATH_PYTHON:-python3}" -m crystalpath
