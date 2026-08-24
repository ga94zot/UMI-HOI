"""Strip object-noun ids from LLaVA answer files, keeping verb ids only.

llava/predict.py extends `action_list` (117 verbs) with `object_list`, so line 2 of each
answer .txt can contain ids >= 117 (object nouns). hicodet/hicodet.py:128 feeds line 2 to the
model as verb tokens, so a verb-only copy is written to a new folder; the source is untouched.
"""
import argparse
import os


def process_file(src_path, dst_path, num_verbs):
    with open(src_path) as f:
        lines = f.read().split("\n")
    names = lines[0].split() if len(lines) > 0 else []
    ids = [int(x) for x in lines[1].split()] if len(lines) > 1 and lines[1].strip() else []
    rest = lines[2:] if len(lines) > 2 else []

    if len(names) == len(ids):
        kept = [(n, i) for n, i in zip(names, ids) if i < num_verbs]
        names_out = [n for n, _ in kept]
    else:
        # names/ids misaligned in the source; ids are what the reader uses, so filter ids only
        kept = [(None, i) for i in ids if i < num_verbs]
        names_out = names
    ids_out = [i for _, i in kept]

    with open(dst_path, "w") as f:
        f.write(" ".join(names_out) + " \n")
        f.write(" ".join(str(i) for i in ids_out) + " \n")
        f.write("\n".join(rest))
    return len(ids) - len(ids_out), len(ids_out) == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--num-verbs", type=int, default=117)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    args = p.parse_args()

    for split in args.splits:
        src_dir = os.path.join(args.src, split)
        dst_dir = os.path.join(args.dst, split)
        os.makedirs(dst_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))
        removed = emptied = 0
        for fn in files:
            r, e = process_file(os.path.join(src_dir, fn), os.path.join(dst_dir, fn), args.num_verbs)
            removed += r
            emptied += e
        print(f"{split}: {len(files)} files, {removed} object ids removed, "
              f"{emptied} files left with no verb id (reader falls back to id 57)")


if __name__ == "__main__":
    main()
