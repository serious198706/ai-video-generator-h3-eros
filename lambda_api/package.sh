#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
build="$(mktemp -d)"
trap 'rm -rf -- "$build"' EXIT
python3 -m pip install -r lambda_api/requirements.txt --target "$build" --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --only-binary=:all:
mkdir -p "$build/h3_service/contracts" "$build/h3_service/media" "$build/h3_service/net" "$build/lambda_api"
cp h3_service/{__init__,config}.py "$build/h3_service/"
cp h3_service/contracts/*.py "$build/h3_service/contracts/"
cp h3_service/media/{__init__,cloud}.py "$build/h3_service/media/"
cp h3_service/net/*.py "$build/h3_service/net/"
cp lambda_api/*.py "$build/lambda_api/"
mkdir -p dist
archive="$PWD/dist/h3-lambda.zip"
rm -f -- "$archive"
(cd "$build" && zip -qr "$archive" .)
echo "$archive"
