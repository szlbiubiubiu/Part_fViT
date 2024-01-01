# Part-based Face Recognition with Vision Transformers

This is the Pytorch implementation project of our BMVC 2022 paper

>[Part-based Face Recognition with Vision Transformers ](https://bmvc2022.mpi-inf.mpg.de/0611.pdf). 
><br>Zhonglin Sun, Georgios Tzimiropoulos<br>



Our code is partly borrowed from Face Transformer for Recognition (https://github.com/zhongyy/Face-Transformer) and Insightface(https://github.com/deepinsight/insightface).

## To Do
- [x] [baseline fViT](vit_pytorch_my/vit_face_nolandmark)
- [x] [Part fViT](vit_pytorch_my/vit_face_withlandmark)
- [x] Checkpoints: [Part fViT](https://drive.google.com/file/d/1ev-y0aOmt1mhQCCZwh3ef204ibszi1Rl/view?usp=sharing) (Performance on IJB-C: TAR@FAR=1e-4 97.29)
- [ ] Training scripts




## Usage
1.fViT
```
import torch
from vit_pytorch_my.vit_face_nolandmark import ViT_face_landmark_patch8

HEAD_NAME='CosFace'
num_patches=196
patch_size=8
with_land=False

backbone=ViT_face_landmark_patch8(
                        loss_type = HEAD_NAME,
                        GPU_ID = None,
                        num_class = NUM_CLASS,
                        num_patches=num_patches,
                        image_size=112,
                        patch_size=patch_size,#8
                        dim=768,#512
                         depth=12,#20
                         heads=11,#8
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0.1,
                        with_land=with_land
                    )

input_data=torch.rand(1,3,112,112)
embeddings= backbone(input_data)
```

2.Part fViT
```
import torch
from vit_pytorch_my.vit_face_withlandmark import ViT_face_landmark_patch8

HEAD_NAME='CosFace'
num_patches=196  
patch_size=8
with_land=True

backbone=ViT_face_landmark_patch8(
                        loss_type = HEAD_NAME,
                        GPU_ID = None,
                        num_class = NUM_CLASS,
                        num_patches=num_patches,
                        image_size=112,
                        patch_size=patch_size,#8
                        dim=768,#512
                        depth=12,#20
                        heads=11,#8
                        mlp_dim=2048,
                        dropout=0.1,
                        emb_dropout=0.1,
                        with_land=with_land
                    )

input_data=torch.rand(1,3,112,112)
embeddings= backbone(input_data)
```


## Citation
```
@inproceedings{Sun_2022_BMVC,
author    = {Zhonglin Sun and Georgios Tzimiropoulos},
title     = {Part-based Face Recognition with Vision Transformers},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0611.pdf}
}
```


