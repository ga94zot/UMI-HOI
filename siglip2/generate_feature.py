"""Precompute SigLIP 2 visual tokens for one image folder.

Writes <out>/<image_stem>.pt = Tensor(577, 1536) fp32 for google/siglip2-giant-opt-patch16-384:
rows 0..575 are the 24x24 patch tokens (last_hidden_state), row 576 is the pooled image token.
Requires a transformers version with SigLIP 2 support (>= 4.49) and the patched
`get_image_features` from modified_forward.py, which returns (last_hidden_state, pooled_output).

Example:
    python siglip2/generate_feature.py --images hicodet/hico_20160224_det/images/train2015 \
        --out /data/hico_siglip2_feature/train --device cuda:0
"""
import argparse
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--images", required=True, help="folder with the .jpg images of one split")
parser.add_argument("--out", required=True, help="output folder for the per-image .pt files")
parser.add_argument("--ckpt", default="google/siglip2-giant-opt-patch16-384")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

# Release cleanup 2026-09-09: paths and the GPU used to be hard-coded (`/path/to/...`
# placeholders, `cuda:1`); they are now command-line arguments and existing outputs are skipped
# so an interrupted run can be resumed.
# folder_dir = "/path/to/vcoco/mscoco2014/val2014/"
# output_dir = "/path/to/vcoco_siglip_feature/val/"
# ckpt = "google/siglip2-giant-opt-patch16-384"
# model = AutoModel.from_pretrained(ckpt, device_map="cuda:1").eval()
os.makedirs(args.out, exist_ok=True)
model = AutoModel.from_pretrained(args.ckpt, device_map=args.device).eval()
processor = AutoProcessor.from_pretrained(args.ckpt)
image_list = sorted(os.listdir(args.images))
total_num = len(image_list)
for idx, img_file in enumerate(image_list):
    out_path = os.path.join(args.out, os.path.splitext(img_file)[0] + ".pt")
    if os.path.exists(out_path):
        continue
    print(f"processing: {img_file}, [{idx}/{total_num}]")
    image = Image.open(os.path.join(args.images, img_file)).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    # inputs.data['pixel_values'] = inputs.data['pixel_values'].to('cuda:1')
    inputs.data['pixel_values'] = inputs.data['pixel_values'].to(args.device)
    with torch.no_grad():
        last_hs, pooled_output = model.get_image_features(**inputs)
        last_hs = last_hs.squeeze(0)
        feature = torch.cat([last_hs, pooled_output], dim=0)
        torch.save(feature.cpu(), out_path)
