#!/usr/bin/env bash
# Install a LiDAR-only config overlay into a checkout of jorichee14/incop_analysis.
#
#   ./install_overlay.sh incop <incop_checkout> <incop_dataset_dir> [scene]
#   ./install_overlay.sh opv2v <incop_checkout> <opv2v_root> [<opencood_checkout>]
#
# e.g.  ./install_overlay.sh incop ~/incop_analysis ~/InCoP/dataset hospital
#       ./install_overlay.sh opv2v ~/incop_analysis ~/cpfa/data/OPV2V ~/cpfa/OpenCOOD
#
# The overlay is deliberately small: InCoP's model already runs LiDAR-only on both
# datasets unmodified (see make_configs.py for why). Only the opv2v arm needs a file
# copied, because its anchor-based head wants point_pillar_loss and InCoP ships only
# center_head_loss. The incop arm needs nothing but configs.
set -euo pipefail

PRESET="${1:?usage: install_overlay.sh <incop|opv2v> <incop_checkout> <data_root> [...]}"
INCOP="${2:?missing incop checkout}"
DATA="${3:?missing data root}"
EXTRA="${4:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -d "$INCOP/opencood" ] || { echo "not an InCoP checkout: $INCOP" >&2; exit 1; }

case "$PRESET" in
  incop)
    SCENE="${EXTRA:-hospital}"
    OUT="$INCOP/opencood/hypes_yaml/incop_lidar_$SCENE"
    python "$HERE/make_configs.py" --preset incop --data-root "$DATA" \
      --scene "$SCENE" --out "$OUT" --methods ours,where2comm,cobevt,v2xvit,ermvp
    LOG_HINT="the indoor arm needs no loss port: center_head_loss already ships"
    ;;
  opv2v)
    OUT="$INCOP/opencood/hypes_yaml/opv2v"
    # anchor_based matches the seven OpenCOOD baselines and lets VoxelPostprocessor plus
    # the parent study's verified compute_metrics apply unchanged -- but it needs a loss
    # InCoP does not ship. create_loss() resolves by importlib on opencood.loss.<name>,
    # so dropping the file in registers it.
    if [ -f "$INCOP/opencood/loss/point_pillar_loss.py" ]; then
      LOG_HINT="point_pillar_loss.py already present"
    elif [ -n "$EXTRA" ] && [ -f "$EXTRA/opencood/loss/point_pillar_loss.py" ]; then
      cp "$EXTRA/opencood/loss/point_pillar_loss.py" "$INCOP/opencood/loss/"
      LOG_HINT="copied point_pillar_loss.py from $EXTRA -- CHECK its expected args against
the loss block in the generated configs; HEAL/CoAlign and DerrickXuNu/OpenCOOD differ"
    else
      LOG_HINT="point_pillar_loss.py NOT installed. Pass an OpenCOOD/HEAL checkout as the
4th argument, or edit the configs to head_type: center_head"
    fi
    python "$HERE/make_configs.py" --preset opv2v --data-root "$DATA" --out "$OUT"
    ;;
  *) echo "unknown preset: $PRESET (want incop or opv2v)" >&2; exit 1 ;;
esac

cat <<EOF

Configs in $OUT
$LOG_HINT

Run order (from \$INCOP):

  1. single-agent pretrain -> encoder_m1 weights AND the ego-only floor
     python opencood/tools/train_isaac.py --hypes_yaml $OUT/single_m1_pointpillar.yaml

  2. regenerate pointing at that run so the fusion configs load the encoder
     python $HERE/make_configs.py --preset $PRESET --data-root $DATA \\
       --out $OUT --pretrained opencood/logs/<run dir>

  3. train each method (cobevt first -- cross-codebase calibration bridge)
     for m in cobevt where2comm ours; do
       python opencood/tools/train_isaac.py --hypes_yaml $OUT/\$m.yaml
     done

  4. clean-channel baseline before any impairment work
     python opencood/tools/inference_isaac.py --model_dir opencood/logs/<run> \\
       --fusion_method intermediate --eval_split test

  5. inertness gate, then the sweep -- see incop_port/README.md
EOF
