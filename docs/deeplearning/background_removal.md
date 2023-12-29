# Background removal tools

### kiuikit

re-wrapped rembg:

```bash
pip install kiui[full]

### cli
python -m kiui.bg --help
python -m kiui.bg input_folder output_folder
python -m kiui.bg input_folder output_folder --return_mask --lcc
```

```python
from kiui.bg import remove
remove_folder(input_folder, output_folder, return_mask=True)
```


### [Carvekit](https://github.com/OPHoperHPO/image-background-remove-tool)

```bash
pip install carvekit

### cli
python -m carvekit --help
python -m carvekit -i input_file -o output_file --device cuda --fp16 True
# output_folder should be created first
python -m carvekit -i input_folder -o output_folder --batch_size 32 --device cuda --fp16 True
```

```python
import os
import tqdm
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch

class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )
    
    @torch.no_grad()
    def __call__(self, image):
        # image: PIL Image
        image = self.interface([image])[0]
        image = np.array(image)
        return image

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)

opt = parser.parse_args()

if opt.path[-1] == '/':
    opt.path = opt.path[:-1]

out_dir = os.path.join(os.path.dirname(opt.path), f'mask')
os.makedirs(out_dir, exist_ok=True)

print(f'[INFO] removing background: {opt.path} --> {out_dir}')

model = BackgroundRemoval()

def run_image(img_path):
    # img: filepath
    image = Image.open(img_path)
    carved_image = model(image) # [H, W, 4]
    mask = (carved_image[..., -1] > 0).astype(np.uint8) * 255 # [H, W]
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
    cv2.imwrite(out_path, mask)

img_paths = glob.glob(os.path.join(opt.path, '*'))
for img_path in tqdm.tqdm(img_paths):
    run_image(img_path)
```


### [Rembg](https://github.com/danielgatis/rembg)

best quality.

```bash
pip install rembg[gpu]

# cli won't work for me... click complains.
rembg i input_file output_file
rembg p input_folder output_folder
```


```python
from rembg import remove
import cv2

input = cv2.imread('input.png')

output = remove(input) # [H, W, 4]
output = remove(input, post_process=True) # apply morphological opening and gaussian blurring
output = remove(input, alpha_matting=True) # alpha matting, better for hair-like object, but need hyperparameter tuning...

cv2.imwrite('output.png', output)
```


### [BackgroundRemover](https://github.com/nadermx/backgroundremover)

better than carvekit, but slightly worse than rembg.

```bash
pip install backgroundremover

# cli
backgroundremover -i "/path/to/image.jpeg" -o "output.png"
```

