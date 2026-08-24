"""Pack per-image SigLIP 2 feature files into safetensors shards and upload them to a
Hugging Face dataset repository.

Source layout (what siglip2/generate_feature.py writes and hicodet.py reads):
    <src>/<split>/<image_stem>.pt   Tensor (577, 1536) fp32

Hub layout produced here:
    <split>/shard-XXXX.safetensors  {image_stem: Tensor(577, 1536) fp32, ...}
    <split>/index.json              {image_stem: "shard-XXXX.safetensors"}
    shards.json                     per-shard sha256 / size / image count
    README.md, generate_feature.py, hf_download_features.py

Shards are packed and uploaded in a two-stage pipeline (at most three shards staged on local
disk: one uploading, one queued, one being packed). Re-running skips shards that already
exist on the hub with the same size, so an interrupted upload can be resumed. Tensors are
stored bit-exact (fp32).

Example (dry run into a private scratch repo, 40 images per shard, 2 shards per split):
    python siglip2/hf_upload_features.py --repo-id <user>/scratch --private \
        --shard-size-gb 0.14 --limit-shards 2
Full run:
    python siglip2/hf_upload_features.py --repo-id Pikaqiu0114/hicodet-siglipv2-features
"""
import argparse
import hashlib
import json
import os
import queue
import random
import shutil
import threading
import time

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file

EXPECTED_SHAPE = (577, 1536)
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sha256_of(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_shards(split_dir, shard_bytes):
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".pt"))
    shards, current, size = [], [], 0
    for f in files:
        fsize = os.path.getsize(os.path.join(split_dir, f))
        if current and size + fsize > shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(f)
        size += fsize
    if current:
        shards.append(current)
    return shards


def pack_shard(split_dir, files, out_path, split, shard_name):
    tensors = {}
    for f in files:
        t = torch.load(os.path.join(split_dir, f), map_location="cpu", weights_only=True)
        if tuple(t.shape) != EXPECTED_SHAPE or t.dtype != torch.float32:
            raise ValueError(f"{f}: unexpected {tuple(t.shape)} {t.dtype}")
        tensors[f[:-3]] = t.contiguous()
    tmp = out_path + ".tmp"
    save_file(tensors, tmp, metadata={"split": split, "shard": shard_name,
                                      "num_images": str(len(files)),
                                      "layout": "rows 0..575 patch tokens (24x24), row 576 pooled token",
                                      "model": "google/siglip2-giant-opt-patch16-384"})
    os.replace(tmp, out_path)
    # spot-check: the written shard must reproduce the source tensors bit for bit
    written = load_file(out_path)
    for f in random.sample(files, min(3, len(files))):
        src = torch.load(os.path.join(split_dir, f), map_location="cpu", weights_only=True)
        if not torch.equal(written[f[:-3]], src):
            raise RuntimeError(f"verification failed for {f} in {out_path}")
    return len(files)


def hub_file_size(api, repo_id, path_in_repo):
    info = api.get_paths_info(repo_id, [path_in_repo], repo_type="dataset")
    return info[0].size if info else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with <split>/<image_stem>.pt files")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--staging", default=os.path.expanduser("~/hf_staging"),
                    help="local scratch folder for packed shards (at most three at a time)")
    ap.add_argument("--shard-size-gb", type=float, default=10.0)
    ap.add_argument("--limit-shards", type=int, default=0, help="debug: only the first N shards per split")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--card", default=os.path.join(HERE, "HF_DATASET_CARD.md"))
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    os.makedirs(args.staging, exist_ok=True)
    shard_bytes = int(args.shard_size_gb * (1 << 30))
    manifest_path = os.path.join(args.staging, "shards.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    for split in args.splits:
        split_dir = os.path.join(args.src, split)
        shards = plan_shards(split_dir, shard_bytes)
        if args.limit_shards:
            shards = shards[:args.limit_shards]
        log(f"{split}: {sum(map(len, shards))} images in {len(shards)} shards")
        index = {}
        todo = queue.Queue(maxsize=1)  # bounded: at most one packed shard waits for upload

        def packer():
            for i, files in enumerate(shards):
                shard_name = f"shard-{i:04d}.safetensors"
                path_in_repo = f"{split}/{shard_name}"
                local = os.path.join(args.staging, f"{split}-{shard_name}")
                for f in files:
                    index[f[:-3]] = shard_name
                if path_in_repo in manifest and hub_file_size(api, args.repo_id, path_in_repo) == manifest[path_in_repo]["size"]:
                    log(f"skip {path_in_repo} (already on hub)")
                    continue
                t0 = time.time()
                n = pack_shard(split_dir, files, local, split, shard_name)
                log(f"packed {path_in_repo}: {n} images, {os.path.getsize(local) / 2**30:.2f} GiB in {time.time() - t0:.0f}s")
                todo.put((path_in_repo, local, n))
            todo.put(None)

        threading.Thread(target=packer, daemon=True).start()
        while True:
            item = todo.get()
            if item is None:
                break
            path_in_repo, local, n = item
            t0 = time.time()
            digest = sha256_of(local)
            api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=args.repo_id,
                            repo_type="dataset", commit_message=f"Add {path_in_repo}")
            manifest[path_in_repo] = {"sha256": digest, "size": os.path.getsize(local), "num_images": n}
            json.dump(manifest, open(manifest_path, "w"), indent=1)
            os.remove(local)
            log(f"uploaded {path_in_repo} in {time.time() - t0:.0f}s")

        index_local = os.path.join(args.staging, f"{split}-index.json")
        json.dump(index, open(index_local, "w"), indent=0)
        api.upload_file(path_or_fileobj=index_local, path_in_repo=f"{split}/index.json",
                        repo_id=args.repo_id, repo_type="dataset", commit_message=f"Add {split}/index.json")

    extras = {manifest_path: "shards.json",
              args.card: "README.md",
              os.path.join(HERE, "generate_feature.py"): "generate_feature.py",
              os.path.join(HERE, "hf_download_features.py"): "hf_download_features.py"}
    for local, remote in extras.items():
        if os.path.exists(local):
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=args.repo_id,
                            repo_type="dataset", commit_message=f"Add {remote}")
    log(f"done: {sum(m['num_images'] for m in manifest.values())} images in {len(manifest)} shards -> {args.repo_id}")


if __name__ == "__main__":
    main()
