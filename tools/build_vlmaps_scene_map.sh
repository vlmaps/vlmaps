#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Uso:"
    echo "  $0 <scenario>"
    echo
    echo "Ejemplo:"
    echo "  $0 boxes_I0"
    exit 1
fi

SCENARIO="$1"

REPO="$HOME/vlmaps_ws/vlmaps"
DATA="$HOME/vlmaps_data_benchmark/$SCENARIO"
BUILD_ROOT="$HOME/vlmaps_build_root"

MAP="$DATA/vlmap_cam/vlmaps_cam.h5df"

if [ ! -d "$DATA" ]; then
    echo "ERROR: dataset no existe:"
    echo "$DATA"
    exit 1
fi

if [ -e "$MAP" ]; then
    echo "ERROR: mapa ya existe:"
    echo "$MAP"
    exit 1
fi

cd "$REPO"

if ! grep -q '^skip_frame: 1$' \
    config/map_config/vlmaps_jackal.yaml
then
    echo "ERROR: skip_frame debe ser 1"
    exit 1
fi

mkdir -p "$BUILD_ROOT"

if [ -L "$BUILD_ROOT/scene" ]; then
    unlink "$BUILD_ROOT/scene"
elif [ -e "$BUILD_ROOT/scene" ]; then
    echo "ERROR: scene existe y no es un enlace simbolico"
    exit 1
fi

ln -s "$DATA" "$BUILD_ROOT/scene"

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate vlmaps

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo
echo "========================================"
echo " BUILDING VLMAP"
echo "========================================"
echo "$SCENARIO"
echo

python application/create_map.py \
    data_paths=jackal_benchmark_build \
    map_config=vlmaps_jackal \
    scene_id=0

if [ ! -s "$MAP" ]; then
    echo "ERROR: VLMap no fue generado"
    exit 1
fi

echo
echo "========================================"
echo " MAPA GENERADO"
echo "========================================"
ls -lh "$MAP"
