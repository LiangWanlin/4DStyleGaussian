# 4DStyleGaussian
The code for "4DStyleGaussian: Zero-shot 4D Style Transfer with Gaussian Splatting"
## Setup
Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.

Tested with Pytorch 1.13.1 and CUDA 11.7.
You can change the pytorch version depending on your local machines.

```
git clone https://github.com/LiangWanlin/4DStyleGaussian/
cd 4DStyleGaussian
conda create -n 4DStyleGaussian python=3.8
conda activate 4DStyleGaussian

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Data Preparation
### [DyNeRF](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0)
Download the dataset from [DyNeRF](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) and put scene into `/data/${scene_name}`
Preprocess the data following [4DGaussian](https://github.com/hustvl/4DGaussians)
```
Preparing the data on the "sear_steak" scene
# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/sear_steak
# Second, generate point clouds from input data.
bash colmap.sh data/sear_steak llff
# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py data/sear_steak/colmap/dense/workspace/fused.ply data/sear_steak/points3D_downsample2.ply
```

## Testing
Download the pre-trained model from [Weights](https://drive.google.com/drive/folders/18ckg32l0uG7CLhxo1uhaWT49WY6LYt5n?usp=sharing) and decompressed it into `output/dynerf/`(e.g. `output/dynerf/sear_steak_style`).

Put the style images for testing into `test_style_image`

Test "sear_steak" scene:
```
CUDA_VISIBLE_DEVICES=0 python render_style.py \
    -s "data/sear_steak/" \
    --model_path "output/dynerf/sear_steak_style" \
    --configs "arguments/dynerf/sear_steak.py" \
    --skip_video \
    --skip_train \
    --CapVstNet_path "output/dynerf/sear_steak_style/vstnet/vstnet_15000.th" \
    --gaussian_ply_ckpt "output/dynerf/sear_steak_style/point_cloud/iteration_15000/point_cloud.ply" \
    --gaussian_ckpt "output/dynerf/sear_steak_style/point_cloud/iteration_15000/" \
    --test_style_path  "test_style_image"
```
The stylization results can be found in `output/dynerf/sear_steak_style/test`. 
