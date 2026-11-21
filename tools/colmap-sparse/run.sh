#!/bin/sh
set -eu
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
exec uv run --no-project --with-requirements "$here/../requirements-colmap.txt" --with 'Pillow==12.1.0' python "$here/register.py" "$@"
