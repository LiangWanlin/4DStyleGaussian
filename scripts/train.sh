

CUDA_VISIBLE_DEVICES=3 python train_feature.py \
    -s "data/coffee_martini" \
    --port 6017 \
    --expname "dynerf/coffee_martini_feature" \
    --configs "arguments/dynerf/coffee_martini.py"



CUDA_VISIBLE_DEVICES=3 python train_style.py \
    -s "data/coffee_martini" \
    --port 6017 \
    --expname "dynerf/coffee_martini_style" \
    --configs "arguments/dynerf/coffee_martini.py" \
    --gaussian_ply_ckpt "output/dynerf/coffee_martini_feature/point_cloud/iteration_15000/point_cloud.ply" \
    --gaussian_ckpt "output/dynerf/coffee_martini_feature/point_cloud/iteration_15000/" \
    --CapVstNet_path "output/dynerf/coffee_martini_feature/vstnet/vstnet_15000.th" \
    --style_dir "style_imgs" 


CUDA_VISIBLE_DEVICES=3 python train_style.py \
    -s "/DATA20T/bip/lwl/code/LED-Gaussians/data/dynerf/cut_roasted_beef/" \
    --port 6017 \
    --expname "dynerf/cut_roasted_beef_style" \
    --configs "arguments/dynerf/cut_roasted_beef.py" \
    --gaussian_ply_ckpt "output/dynerf/cut_roasted_beef_feature/point_cloud/iteration_12000/point_cloud.ply" \
    --gaussian_ckpt "output/dynerf/cut_roasted_beef_feature/point_cloud/iteration_12000/" \
    --CapVstNet_path "output/dynerf/cut_roasted_beef_feature/vstnet/vstnet_12000.th" \
    --style_dir "/DATA20T/bip/lwl/data/style_imgs_20240314/" 

CUDA_VISIBLE_DEVICES=3 python train_style.py \
    -s "/DATA20T/bip/lwl/code/LED-Gaussians/data/dynerf/flame_steak/" \
    --port 6017 \
    --expname "dynerf/flame_steak_style" \
    --configs "arguments/dynerf/flame_steak.py" \
    --gaussian_ply_ckpt "output/dynerf/flame_steak_feature/point_cloud/iteration_12000/point_cloud.ply" \
    --gaussian_ckpt "output/dynerf/flame_steak_feature/point_cloud/iteration_12000/" \
    --CapVstNet_path "output/dynerf/flame_steak_feature/vstnet/vstnet_12000.th" \
    --style_dir "/DATA20T/bip/lwl/data/style_imgs_20240314/" 


CUDA_VISIBLE_DEVICES=4 python train_feature.py \
    -s "/DATA20T/bip/lwl/code/LED-Gaussians/data/dynerf/cook_spinach/" \
    --port 6016 \
    --expname "dynerf/cook_spinach_feature" \
    --configs "arguments/dynerf/cook_spinach.py"

CUDA_VISIBLE_DEVICES=3 python train_feature.py \
    -s "/DATA20T/bip/lwl/code/LED-Gaussians/data/dynerf/flame_salmon_1/" \
    --port 6017 \
    --expname "dynerf/flame_salmon_1_feature" \
    --configs "arguments/dynerf/flame_salmon_1.py"