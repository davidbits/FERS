#!/bin/bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly REPO_ROOT="${SCRIPT_DIR}/.."
readonly COVERAGE_DIR="${REPO_ROOT}/build/coverage"
# lcov suppresses a warning category only when the category appears twice.
readonly LCOV_IGNORE_ERRORS="source,gcov,negative,inconsistent,inconsistent"
readonly GENHTML_IGNORE_ERRORS="source,inconsistent,inconsistent"

TESTS_ONLY=false

# --- argument parsing ---
for arg in "$@"; do
	case "$arg" in
		--tests-only)
			TESTS_ONLY=true
			;;
		*)
			die "unknown argument: $arg"
			;;
	esac
done

die() {
	printf 'Error: %s\n' "$*" >&2
	exit 1
}

on_err() {
	local exit_code=$?
	printf 'Error: command failed (exit %d) at %s:%s: %s\n' \
		"$exit_code" "${BASH_SOURCE[0]}" "${BASH_LINENO[0]}" "${BASH_COMMAND}" >&2
	exit "$exit_code"
}

require_cmd() {
	command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

trap on_err ERR

# Only require lcov/genhtml if we actually use them
for cmd in cmake ctest; do
	require_cmd "$cmd"
done

if [ "$TESTS_ONLY" = false ]; then
	for cmd in lcov genhtml; do
		require_cmd "$cmd"
	done
fi

cd "$REPO_ROOT"

cmake --preset=coverage
cmake --build --preset=coverage --parallel
ctest --preset=coverage --parallel

# --- early exit if tests-only ---
if [ "$TESTS_ONLY" = true ]; then
	exit 0
fi

cd "$COVERAGE_DIR"

lcov --capture --directory . --output-file coverage.info \
	--rc geninfo_unexecuted_blocks=1 \
	--ignore-errors "$LCOV_IGNORE_ERRORS"
lcov --remove coverage.info '/usr/*' '*/tests/*' '*/vcpkg_installed/*' --output-file coverage_filtered.info \
	--ignore-errors "$LCOV_IGNORE_ERRORS"
genhtml coverage_filtered.info --output-directory coverage_report \
	--ignore-errors "$GENHTML_IGNORE_ERRORS"
