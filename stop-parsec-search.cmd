@echo off
pushd "%~dp0"
docker compose -f compose.yaml -f compose.gpu.yaml down
popd
