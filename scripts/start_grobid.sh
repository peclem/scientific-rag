#!/usr/bin/env bash
# Start GROBID locally in Docker for PDF parsing.
#
# We pin a specific version (0.8.1) for reproducibility. This is the full
# image; the first pull is a few GB. GROBID runs its CRF models on CPU by
# default, which is what we want here: the GPU is reserved for the LLM, and
# CRF parsing is plenty good for our needs.
#
# Usage:
#   ./scripts/start_grobid.sh          # start (or reuse) the container
#   docker logs -f grobid              # watch startup
#   docker stop grobid                 # stop it

set -euo pipefail

IMAGE="grobid/grobid:0.8.1"
NAME="grobid"
PORT="8070"

if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "GROBID already running on port ${PORT}."
  exit 0
fi

# Reuse a stopped container if it exists, otherwise create a fresh one.
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Starting existing ${NAME} container..."
  docker start "${NAME}"
else
  echo "Creating ${NAME} container from ${IMAGE} (first pull may be large)..."
  docker run -d --name "${NAME}" -p "${PORT}:8070" --init "${IMAGE}"
fi

echo -n "Waiting for GROBID to become healthy"
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/api/isalive" | grep -qi true; then
    echo " ... up."
    exit 0
  fi
  echo -n "."
  sleep 2
done

echo
echo "GROBID did not report healthy in time. Check: docker logs ${NAME}"
exit 1
