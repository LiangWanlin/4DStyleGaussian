

CUDA_VISIBLE_DEVICES=3 python render_style.py \
    -s "data/coffee_martini/" \
    --model_path "output/dynerf/coffee_martini_style" \
    --configs "arguments/dynerf/coffee_martini.py" \
    --skip_video \
    --skip_train \
    --CapVstNet_path "output/dynerf/coffee_martini_feature/vstnet/last.th" \
    --gaussian_ply_ckpt "output/dynerf/${scene_name}_feature/point_cloud/iteration_15000/point_cloud.ply" \
    --gaussian_ckpt "output/dynerf/${scene_name}_feature/point_cloud/iteration_15000/" \
    --test_style_path "test_style_image"


