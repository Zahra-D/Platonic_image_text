"""Machine-specific data roots, overridable through the environment.

    CLEVR_DATA  directory holding the dataset manifests (platonic_* folders)
    CLEVR_GEN   clevr-dataset-gen image_generation folder (caption generator)

Unset, both default to the development machine's paths, so local runs are unchanged.
Training configs reference ${CLEVR_DATA}; see CLUSTER.md.
"""
import os

CLEVR_DATA_ROOT = os.environ.setdefault("CLEVR_DATA", "/home/zd25e122/clevr-dataset-gen_clone/output")
CLEVR_GEN_ROOT = os.environ.setdefault("CLEVR_GEN", "/home/zd25e122/clevr-dataset-gen_clone/image_generation")
