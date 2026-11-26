#!/bin/sh
set -eu
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
# pycolmap 4.2.1 publishes wheels for CPython 3.10-3.14 only. Without a bound, uv picks the
# newest installed interpreter (3.15 once present), fails to resolve, and every native run
# silently falls back to Rust camera registration.
exec uv run --no-project --python '>=3.10,<3.15' --with-requirements "$here/../requirements-colmap.txt" --with 'Pillow==12.1.0' python "$here/register.py" "$@"
