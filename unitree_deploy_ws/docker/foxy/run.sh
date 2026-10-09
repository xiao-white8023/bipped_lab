#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_root="$(cd "${script_dir}/../../.." && pwd)"
xauthority_path="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set; start this script from the graphical desktop session." >&2
  exit 1
fi

if [[ ! -r "${xauthority_path}" ]]; then
  echo "Xauthority file is not readable: ${xauthority_path}" >&2
  exit 1
fi

docker_args=(
  --rm
  --interactive
  --tty
  --name g1-foxy
  --network host
  --ipc host
  --env "DISPLAY=${DISPLAY}"
  --env XAUTHORITY=/tmp/host-xauthority
  --env QT_X11_NO_MITSHM=1
  --env ROS_DOMAIN_ID=0
  --env ROS_LOCALHOST_ONLY=0
  --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  --env 'CYCLONEDDS_URI=<CycloneDDS><Domain><General><NetworkInterfaceAddress>enp4s0</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
  --volume "${xauthority_path}:/tmp/host-xauthority:ro"
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw
  --volume "${lab_root}:${lab_root}:rw"
  --workdir "${lab_root}/unitree_deploy_ws"
)


exec docker run "${docker_args[@]}" g1-foxy:local bash
