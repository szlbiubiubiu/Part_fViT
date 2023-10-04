# Part fViT

This is the Pytorch implementation project of our BMVC 2022 paper

>[Part-based Face Recognition with Vision Transformers ](https://bmvc2022.mpi-inf.mpg.de/0611.pdf). 
><br>Zhonglin Sun, Georgios Tzimiropoulos<br>



Our code is partly borrowed from Face Transformer for Recognition (https://github.com/zhongyy/Face-Transformer) and Insightface(https://github.com/deepinsight/insightface).

## To Do
- [x] [baseline fViT](vit_pytorch_my/vit_face_nolandmark)
- [ ] [Part fViT]
- [ ] [Checkpoints]
- [ ] [Training scripts]
<!-- - [x] [ArcFace_mxnet (CVPR'2019)](recognition/arcface_mxnet) -->

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



## Citation
```
@article{sun2022part,
  title={Part-based Face Recognition with Vision Transformers},
  author={Sun, Zhonglin and Tzimiropoulos, Georgios},
  year={2022}
}
```


