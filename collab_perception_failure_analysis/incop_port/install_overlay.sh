#!/usr/bin/env bash
# Install the OPV2V overlay into a checkout of jorichee14/incop_analysis.
#
# The overlay is deliberately small: InCoP's model already runs on OPV2V unmodified
# (see make_opv2v_configs.py for why), so this only adds configs plus the one loss
# function InCoP does not ship.
#
#   ./install_overlay.sh <incop_checkout> <opv2v_root> [<opencood_or_heal_checkout>]
#
# e.g.  ./install_overlay.sh ~/incop_analysis ~/cpfa/data/OPV2V ~/cpfa/OpenCOOD
set -euo pipefail

INCOP="${1:?usage: install_overlay.sh <incop_checkout> <opv2v_root> [<opencood_checkout>]}"
OPV2V="${2:?missing opv2v root (the dir containing train/ validate/ test/)}"
SRC="${3:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -d "$INCOP/opencood" ] || { echo "not an InCoP checkout: $INCOP" >&2; exit 1; }

# ---- 1. point_pillar_loss.py -------------------------------------------------
# InCoP ships only center_head_loss. The anchor_based head (which matches the seven
# baselines and lets VoxelPostprocessor + the parent study's verified compute_metrics
# apply unchanged) needs point_pillar_loss. create_loss() resolves it by importlib on
# opencood.loss.<core_method>, so dropping the file in is enough to register it.
if [ -f "$INCOP/opencood/loss/point_pillar_loss.py" ]; then
  echo "point_pillar_loss.py already present, leaving it alone"
elif [ -n "$SRC" ] && [ -f "$SRC/opencood/loss/point_pillar_loss.py" ]; then
  cp "$SRC/opencood/loss/point_pillar_loss.py" "$INCOP/opencood/loss/"
  echo "copied point_pillar_loss.py from $SRC"
  echo "  !! check its expected args against the loss block in the generated configs."
  echo "     HEAL/CoAlign and DerrickXuNu/OpenCOOD use different schemas."
else
  echo "point_pillar_loss.py NOT installed (no source checkout given)." >&2
  echo "Either pass a third argument pointing at an OpenCOOD/HEAL checkout, or" >&2
  echo "regenerate the configs with --head center_head to use the loss InCoP ships." >&2
fi

# ---- 2. configs --------------------------------------------------------------
python "$HERE/make_opv2v_configs.py" \
  --opv2v-root "$OPV2V" \
  --out "$INCOP/opencood/hypes_yaml/opv2v" \
  "${@:4}"

cat <<EOF

Overlay installed. Run order (all from \$INCOP):

  1. single-agent pretrain -> gives encoder_m1 and the ego-only floor
     python opencood/tools/train_isaac.py --hypes_yaml opencood/hypes_yaml/opv2v/single_m1_pointpillar.yaml

  2. regenerate the fusion configs pointing at that run, so they load the encoder
     python $HERE/make_opv2v_configs.py --opv2v-root $OPV2V \\
       --out \$INCOP/opencood/hypes_yaml/opv2v --pretrained opencood/logs/<run dir>

  3. train each method (cobevt first -- it is the cross-codebase calibration bridge)
     for m in cobevt where2comm ours; do
       python opencood/tools/train_isaac.py --hypes_yaml opencood/hypes_yaml/opv2v/\$m.yaml
     done

  4. clean-channel baseline, confirm sane AP before any impairment work
     python opencood/tools/inference_isaac.py --model_dir opencood/logs/<run> \\
       --fusion_method intermediate --eval_split test

  5. inertness gate, then the sweep -- see incop_port/README.md
EOF
