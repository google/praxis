#!/usr/bin/env bash

set -e

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

PIP_FILE_PREFIX="praxis/pip_package/"
ROOT_DIR="."
export PYTHON_VERSION="${PYTHON_VERSION:-3}"
export PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}"
export DEST="${WHEEL_FOLDER:-/tmp/wheels}"

if [[ -z "${PYTHON_MINOR_VERSION}" ]]; then
  PYTHON="python${PYTHON_VERSION}"
else
  PYTHON="python${PYTHON_VERSION}.${PYTHON_MINOR_VERSION}"
fi

function main() {
  cp ${PIP_FILE_PREFIX}setup.py "${ROOT_DIR}"
  cp ${PIP_FILE_PREFIX}requirements.in "${ROOT_DIR}"
  ${PYTHON} setup.py bdist_wheel

  if [ ! -d "${DEST}" ]; then
    mkdir -p "${DEST}"
  fi

  cp dist/*.whl "${DEST}"
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
