#!/usr/bin/env bash
# Builds the PCRE2 release pinned by SPEC.md §2 (Unicode 16.0) as a static library.
# Used by CI, where the distro package is too old. Usage: build_pcre2.sh <install-prefix>
set -euo pipefail

VERSION=10.45
PREFIX=${1:?usage: build_pcre2.sh <install-prefix>}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

curl -L --fail -o "$WORK/pcre2.tar.bz2" \
  "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-${VERSION}/pcre2-${VERSION}.tar.bz2"
tar -xjf "$WORK/pcre2.tar.bz2" -C "$WORK"
cmake -S "$WORK/pcre2-${VERSION}" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_SHARED_LIBS=OFF -DPCRE2_SUPPORT_JIT=ON \
  -DPCRE2_BUILD_PCRE2GREP=OFF -DPCRE2_BUILD_TESTS=OFF
cmake --build "$WORK/build" -j
cmake --install "$WORK/build"
