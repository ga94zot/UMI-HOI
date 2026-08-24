"""Download the SigLIP 2 HICO-DET feature shards from the Hugging Face hub and unpack them
into the per-image layout that hicodet.py expects at --llava-token-path:

    <out>/<split>/<image_stem>.pt   Tensor (577, 1536) fp32

Example:
    python siglip2/hf_download_features.py --repo-id Pikaqiu0114/hicodet-siglipv2-features \
        --out /data/hico_siglip2_feature
Then train with `--llava-token-path /data/hico_siglip2_feature`.
"""
import argparse
import hashlib
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def sha256_of(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--keep-shards", action="store_true", help="do not delete a shard after unpacking it")
    ap.add_argument("--no-verify", action="store_true", help="skip the sha256 check against shards.json")
    args = ap.parse_args()

    repo = snapshot_download(args.repo_id, repo_type="dataset",
                             allow_patterns=[f"{s}/*" for s in args.splits] + ["shards.json"])
    manifest = json.load(open(os.path.join(repo, "shards.json")))
    for split in args.splits:
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        shard_dir = os.path.join(repo, split)
        for name in sorted(f for f in os.listdir(shard_dir) if f.endswith(".safetensors")):
            path = os.path.join(shard_dir, name)
            if not args.no_verify:
                expected = manifest[f"{split}/{name}"]["sha256"]
                if sha256_of(path) != expected:
                    raise RuntimeError(f"checksum mismatch for {split}/{name}")
            with safe_open(path, framework="pt") as f:
                for stem in f.keys():
                    torch.save(f.get_tensor(stem), os.path.join(out_dir, stem + ".pt"))
            print(f"unpacked {split}/{name}", flush=True)
            if not args.keep_shards:
                os.remove(os.path.realpath(path))
        print(f"{split}: {len(os.listdir(out_dir))} files in {out_dir}")


if __name__ == "__main__":
    main()
