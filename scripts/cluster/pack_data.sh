#!/bin/bash
# Pack everything the cluster needs besides the git repo (~7.5 GB) into one tarball.
#
#   bash scripts/cluster/pack_data.sh /path/to/clevr_cluster_data.tar      # on the dev machine
#   # copy the tar to the cluster, then there:
#   tar -xf clevr_cluster_data.tar -C $SOMEWHERE
#   export CLEVR_DATA=$SOMEWHERE/clevr_data CLEVR_GEN=$SOMEWHERE/image_generation
#   cp -r $SOMEWHERE/repo_outputs/* <repo>/outputs/      # token cache, VQ-VAE, eval triples
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?usage: pack_data.sh OUTPUT.tar}
SRC=${CLEVR_DATA:-/home/zd25e122/clevr-dataset-gen_clone/output}
GEN=${CLEVR_GEN:-/home/zd25e122/clevr-dataset-gen_clone/image_generation}
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
mkdir -p $STAGE/clevr_data/platonic_clevr_v1_5M_train_gpu_visible $STAGE/clevr_data/platonic_text_only_v1_2m \
         $STAGE/clevr_data/platonic_text_only_v1_1m $STAGE/repo_outputs/image_only_2_5m_token_cache \
         $STAGE/repo_outputs/vqvae_training_bs128 $STAGE/repo_outputs/image_eval_triples
ln -s $SRC/platonic_clevr_v1_5M_train_gpu_visible/train_image_only_2_5m.jsonl $STAGE/clevr_data/platonic_clevr_v1_5M_train_gpu_visible/
ln -s $SRC/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl        $STAGE/clevr_data/platonic_clevr_v1_5M_train_gpu_visible/
ln -s $SRC/platonic_text_only_v1_2m/train_text_only_human.jsonl               $STAGE/clevr_data/platonic_text_only_v1_2m/
ln -s $SRC/platonic_text_only_v1_1m/val_text_only_human.jsonl                 $STAGE/clevr_data/platonic_text_only_v1_1m/
ln -s $PWD/outputs/image_only_2_5m_token_cache/train_tokens.pt $STAGE/repo_outputs/image_only_2_5m_token_cache/
ln -s $PWD/outputs/image_only_2_5m_token_cache/val_tokens.pt   $STAGE/repo_outputs/image_only_2_5m_token_cache/
ln -s $PWD/outputs/vqvae_training_bs128/best.pt                $STAGE/repo_outputs/vqvae_training_bs128/
ln -s $PWD/outputs/image_eval_triples/triples_tokens.pt        $STAGE/repo_outputs/image_eval_triples/
ln -s $GEN $STAGE/image_generation
tar -chf "$OUT" -C $STAGE clevr_data repo_outputs image_generation   # -h: store the files, not the links
ls -lh "$OUT"
