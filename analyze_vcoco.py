"""Histogram of the number of distinct actions per V-COCO image."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

REPO = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--anno", default=os.path.join(REPO, "vcoco/instances_vcoco_test.json"),
                    help="instances_vcoco_trainval.json or instances_vcoco_test.json")
parser.add_argument("--out", default="analyze_vcoco_test.png", help="output figure")
args = parser.parse_args()

# Release cleanup 2026-09-09: the annotation and figure paths were hard-coded to a
# previous machine; they are now command-line arguments.
# train_anno_file = json.load(open("/home/MasterThesis/pvic/vcoco/instances_vcoco_trainval.json", "r"))
# test_anno_file = json.load(open("/home/MasterThesis/pvic/vcoco/instances_vcoco_test.json", "r"))
test_anno_file = json.load(open(args.anno, "r"))
verb_number_unique = 0
verb_number_max = 0
verb_hist = [0 for _ in range(10)]
valid_samples = 0
for anno in test_anno_file['annotations']:

    if  len(anno['boxes_h']) == 0:
        continue
    valid_samples += 1
    verb_ids = torch.tensor(anno['actions']).unique().tolist()
    verb_number_unique += len(verb_ids)
    if len(verb_ids) >= verb_number_max:
        verb_number_max = len(verb_ids)
    verb_hist[len(verb_ids) - 1] += 1
print(f"total verb_number_unique={verb_number_unique}, \
      average verb_number_unique={verb_number_unique/valid_samples}, \
      verb_number_max={verb_number_max}")

plt.bar([i+1 for i in range(verb_number_max)],  verb_hist[:verb_number_max])
# plt.savefig("/home/MasterThesis/pvic/analyze_vcoco_train.png")
# plt.savefig("/home/MasterThesis/pvic/analyze_vcoco_test.png")
plt.savefig(args.out)
plt.close()
