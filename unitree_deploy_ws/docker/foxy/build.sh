#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build \
  --network host \
  --build-arg "HTTP_PROXY=${HTTP_PROXY:-}" \
  --build-arg "HTTPS_PROXY=${HTTPS_PROXY:-}" \
  --build-arg "NO_PROXY=${NO_PROXY:-}" \
  --build-arg "http_proxy=${http_proxy:-}" \
  --build-arg "https_proxy=${https_proxy:-}" \
  --build-arg "no_proxy=${no_proxy:-}" \
  --build-arg "USER_ID=$(id -u)" \
  --build-arg "GROUP_ID=$(id -g)" \
  --tag g1-foxy:local \
  "${script_dir}"
