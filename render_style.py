from torchvision import transforms as t
from torchvision import transforms
import numpy as np
import torch
from scene import Scene
import os
from PIL import Image
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import models.wct_update_model as update_model
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from scene import GaussianModel
import concurrent.futures
from ReversibleNetwork import RevResNet
from models.wct_matrix4d import MulLayer as MulLayer4d
import models.wct_cspn as model_spn
import glob
import imageio
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


def save_rendered_features(feature_list, path, feature_level):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def save_feature(feature, count, path):
        try:
            save_path = os.path.join(path, '{0:05d}_s{}'.format(count, feature_level) + ".pth")
            torch.save(feature, save_path)
            return count, True
        except:
            return count, False

    for index, feature in enumerate(feature_list):
        # save_path = os.path.join(path, '{0:05d}_s{}'.format(count, feature_level) + ".pth")
        # torch.save(feature, save_path)
        tasks.append(executor.submit(save_feature, feature, index, path))
    executor.shutdown()
    # for index, status in enumerate(tasks):
    #     if status == False:
    #         write_image(image_list[index], index, path)


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, views, gaussians, scene,pipeline, background, cam_type):

    CapVstNet_path=args.CapVstNet_path
    CapVstNet = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=16, sp_steps=2)
    state_dict=torch.load(CapVstNet_path)
    CapVstNet.load_state_dict(state_dict)
    for param in CapVstNet.parameters():
        param.requires_grad = False
    CapVstNet=CapVstNet.to("cuda")
    gt = views[0].original_image[0:3, :, :]
    H=gt.shape[1]
    W=gt.shape[2]
    newH = (H // 8 //8) * 8 * 8
    newW = (W // 8 //8) * 8 * 8

    transforms_style_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    ckpt_matrix = f"{args.model_path}/matrix.th"
    ckpt_spn = f"{args.model_path}/spn.th"
    matrix = MulLayer4d("r31")
    matrix.load_state_dict(torch.load(ckpt_matrix))
    matrix=matrix.to("cuda")
    spn = model_spn.resnet50(pretrained = False)
    best_model_dict = torch.load(ckpt_spn)
    best_model_dict = update_model.remove_moudle(best_model_dict)
    spn.load_state_dict(update_model.update_model(spn, best_model_dict))
    spn=spn.to("cuda")

    style_img_dir=args.test_style_path
    
    style_imgs_paths = glob.glob(os.path.join(style_img_dir, '*'))
    viewpoint_cam=views[0]
    render_pkg = render(viewpoint_cam, gaussians, pipeline, background, cam_type=cam_type)
    canonical_features= render_pkg["colors_precomp"]
    for style_img_path in style_imgs_paths:
        style_name=os.path.basename(style_img_path)
        img = Image.open(style_img_path)
        img_t = transforms_style_img(img)
        img_t = img_t.unsqueeze(dim=0).to("cuda")  
        render_path = os.path.join(model_path, name,str(style_name), "renders")
        gts_path = os.path.join(model_path, name ,str(style_name),"gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        render_images = []
        propagated_images=[]
        transfer_images = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            render_pkg = render(view, gaussians, pipeline, background, cam_type=cam_type)
            rendering = render_pkg["render"].detach() 
            rendered_features = render_pkg["rendered_feature"].unsqueeze(0).detach()  
            render_images.append(to8b(rendering).transpose(1,2,0))
            
            if name in ["train", "test"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt  = view['image']
                

            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + "_rendered.png"))
       
            sF = CapVstNet(img_t,forward=True) 
            cF = rendered_features.detach() 

            # WCT
            feature, transmatrix = matrix(canonical_features, cF, sF)
            feature=t.functional.resize(feature, size=(feature.shape[2]//16*16,feature.shape[3]//16*16), interpolation=t.InterpolationMode.BILINEAR)
            
            transfer = CapVstNet(feature,forward=False)  
            transfer = transfer.clamp(0, 1)

            cF=t.functional.resize(cF, size=(cF.shape[2]//16*16,cF.shape[3]//16*16), interpolation=t.InterpolationMode.BILINEAR)
            rec = CapVstNet(cF,forward=False)  
            
            rgb_map_t = t.functional.resize(rec, size=(newH,newW), interpolation=t.InterpolationMode.BILINEAR)  
            transfer_ = t.functional.resize(transfer, size=(newH,newW), interpolation=t.InterpolationMode.BILINEAR)
           
            propagated = spn(rgb_map_t,transfer_) 
            propagated = t.functional.resize(propagated, size=(transfer.shape[2],transfer.shape[3]), interpolation=t.InterpolationMode.BILINEAR)
            propagated_images.append(to8b(propagated.squeeze()).transpose(1,2,0))
            transfer_images.append(to8b(transfer.squeeze()).transpose(1,2,0))
            torchvision.utils.save_image(propagated, os.path.join(render_path, '{0:05d}'.format(idx) + "_propagated.png"))
            torchvision.utils.save_image(transfer, os.path.join(render_path, '{0:05d}'.format(idx) + "_transfer.png"))
            torchvision.utils.save_image(rec, os.path.join(render_path, '{0:05d}'.format(idx) + "_rec.png"))

        
        imageio.mimwrite(os.path.join(render_path, 'video_propagated.mp4'), propagated_images, fps=30)
        imageio.mimwrite(os.path.join(render_path, 'video_transfer.mp4'), transfer_images, fps=30)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type = scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians.load_ply(args.gaussian_ply_ckpt)
        gaussians.load_model(args.gaussian_ckpt)
        if not skip_train:
            render_set(dataset.model_path, "train", scene.getTrainCameras(), gaussians, scene,pipeline, background, cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.getTestCameras(), gaussians, scene,pipeline, background, cam_type)
        
        if not skip_video:
            render_set(dataset.model_path, "video", scene.getVideoCameras(), gaussians, scene,pipeline, background, cam_type)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--CapVstNet_path", type=str, default ="") 
    parser.add_argument("--gaussian_ply_ckpt", type=str, default ="") 
    parser.add_argument("--gaussian_ckpt", type=str, default ="") 
    parser.add_argument("--test_style_path", type=str, default ="") 
    args = get_combined_args(parser)
    
    print("Rendering " , args.model_path)
    if args.configs:
        # import mmcv
        import mmengine
        from utils.params_utils import merge_hparams
        # config = mmcv.Config.fromfile(args.configs)
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)