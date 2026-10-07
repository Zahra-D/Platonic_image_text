#!/bin/bash
# Build a controlled image-evaluation set: anchor / paraphrase / binding-swap.
# Scenes are generated fresh so they are held out from every training corpus.
set -u
REPO=/home/zd25e122/clevr-dataset-gen_clone
GEN=$REPO/image_generation
B=$REPO/blender/blender-4.5.12-linux-x64/blender
RULES=$REPO/output/platonic_clevr_v1_5M_train_gpu_visible/platonic_rules_v1.json
OUT=${OUT:-/home/zd25e122/clevr_discrete_diffusion/outputs/image_eval_triples}
N=${N:-2000}
GPUS=${GPUS:-"2 3"}
WPG=${WPG:-2}
SEED_GEN=${SEED_GEN:-20261005}
SEED_CAM=${SEED_CAM:-4242}      # anchor + swap share this -> identical camera
SEED_PAR=${SEED_PAR:-7777}      # paraphrase -> different camera
mkdir -p $OUT/logs
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

render(){  # $1=mode(gen|replay) $2=tag $3=seed $4=specfile(or "") 
  local mode=$1 tag=$2 seed=$3 spec=$4
  local workers=() w=0
  for g in $GPUS; do for ((k=0;k<WPG;k++)); do workers+=("$g"); done; done
  local nw=${#workers[@]}
  local per=$(( (N + nw - 1) / nw ))
  log "$tag: $N images, $nw workers, $per each"
  for ((w=0; w<nw; w++)); do
    local start=$((w*per)); local count=$per
    (( start >= N )) && continue
    (( start+count > N )) && count=$((N-start))
    local g=${workers[$w]}
    (
      cd $GEN
      if [ "$mode" = "gen" ]; then
        extra="--world_rules_json $RULES --num_images $count"
      else
        python3 - "$spec" "$start" "$count" "$OUT/shard_${tag}_$w.json" <<'PY'
import json,sys
spec,start,count,out=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
d=json.load(open(spec)); sc=d["scenes"] if isinstance(d,dict) else d
json.dump({"scenes":sc[start:start+count]},open(out,"w"))
PY
        extra="--input_scene_file $OUT/shard_${tag}_$w.json --skip_visibility_check 1"
      fi
      $B --background -noaudio --python render_images.py -- $extra \
        --seed $seed --start_idx $start --split $tag \
        --width 96 --height 64 --min_pixels_per_object 100 \
        --num_digits 6 --images_per_dir 0 \
        --render_num_samples 64 --render_tile_size 64 \
        --use_gpu 1 --gpu_backend OPTIX --gpu_device_index $g \
        --output_image_dir $OUT/$tag --output_scene_dir $OUT/${tag}_scn \
        --output_scene_file $OUT/${tag}_$w.json
    ) > $OUT/logs/${tag}_$w.log 2>&1 &
  done
  wait
  log "$tag done: $(ls $OUT/$tag/*.png 2>/dev/null | wc -l) images"
}

# -- step 0: fresh held-out scenes (their JSONs carry 3d_coords for replay)
if [ "$(ls $OUT/gen/*.png 2>/dev/null | wc -l)" -lt $N ]; then render gen gen $SEED_GEN ""; fi

# -- step 1: anchor + swap specs from those scenes
python3 /home/zd25e122/clevr_discrete_diffusion/scripts/make_triple_specs.py \
    --scene-dir $OUT/gen_scn --out $OUT --num $N >> $OUT/logs/specs.log 2>&1
log "specs: $(python3 -c "import json;print(len(json.load(open('$OUT/anchors.json'))['scenes']))") triples"

# -- steps 2-4: the three controlled renders
render replay anchor $SEED_CAM $OUT/anchors.json
render replay swap   $SEED_PAR $OUT/swaps.json
render replay para   $SEED_PAR $OUT/anchors.json
log "TRIPLES COMPLETE"
