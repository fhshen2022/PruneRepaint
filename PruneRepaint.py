import numpy as np
from PIL import Image
import torch
import time
from RGB_VST.Models.ImageDepthNet import ImageDepthNet
import argparse
from diffusers import DDIMScheduler, AutoencoderKL,StableDiffusionControlNetInpaintPipeline, ControlNetModel
from IP_Adapter.ip_adapter import IPAdapter,IPAdapterPlus
from utils import make_inpaint_condition,cal_saliency_energy, check_generated_image,get_saliency_map,CSC,adaptive_repainting_region_determination
from collections import OrderedDict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--target_ratio', default="16/9", type=str, help='target retargeted aspect ratio')
    parser.add_argument('--save_image_path', default="./output/retargeted_results/retargetme_16_9/", type=str, help='path to save retargeted images')
    parser.add_argument('--input_image_path', default="./input/retargetme/", type=str, help='path to input images')
    parser.add_argument('--save_saliency_path', default="./output/saliency/", type=str, help='path to save saliency')
    parser.add_argument('--img_size', default=224, type=int, help='slaiency detection network input size')
    parser.add_argument('--pretrained_model', default='./RGB_VST/pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load pretrained salient detection backbones')
    parser.add_argument('--save_model_dir', default='./RGB_VST/checkpoint/', type=str, help='load pretrained salient detection models')
    
    parser.add_argument('--ip_path', default="./IP_Adapter/", type=str, help='path to ip adapter folder')
    parser.add_argument('--sd_path', default="runwayml/stable-diffusion-v1-5", type=str, help='path to save sd')
    parser.add_argument('--vae_path', default="stabilityai/sd-vae-ft-mse", type=str, help='path to vae')
    parser.add_argument('--image_encoder_path', default="models/image_encoder/", type=str, help='path to image encoder')
    parser.add_argument('--ip_ckpt', default="models/ip-adapter-plus_sd15.bin", type=str, help='path to ip adapter')
    parser.add_argument('--controlnet_path', default="lllyasviel/control_v11p_sd15_inpaint", type=str, help='path to controlnet')
    
    parser.add_argument('--side_out_path', default="./output/tmp_out/", type=str, help='path to temporal output')
    parser.add_argument('--csc_out_path', default="seamcarving/", type=str, help='path to content aware seam carving output')
    parser.add_argument('--mask_out_path', default="mask_images/", type=str, help='path to seam mask')
    
    args = parser.parse_args()
    
    ######### hyper parameters ###########
    numerator, demoninator = map(int,args.target_ratio.split('/'))
    target_ratio = numerator/demoninator
    cut_saliency_ratio_w = 0.5 if target_ratio>4/3 else 0.3#横向最多允许丢失0.3的显著性区域；当比例过于极端时允许丢失0.5。
    cut_saliency_ratio_h = 0.1
    
    ######### create dirs ########## 
    save_path = args.save_image_path
    saliency_path = args.save_saliency_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(saliency_path):
        os.makedirs(saliency_path)
    if not os.path.exists(args.side_out_path+args.csc_out_path):
        os.makedirs(args.side_out_path+args.csc_out_path)
    if not os.path.exists(args.side_out_path+args.mask_out_path):
        os.makedirs(args.side_out_path+args.mask_out_path)
    
    ########## read data ###########
    image_file = args.input_image_path
    image_name_list = os.listdir(image_file)
    
    ########## load saliency detection model #########
    saliency_net = ImageDepthNet(args).cuda()
    model_path = args.save_model_dir + 'RGB_VST.pth'
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    saliency_net.load_state_dict(new_state_dict)
    print('Salient Detection Model loaded from {}'.format(model_path))
    
    ########### load Adaptive Repainting model ##########
    ip_path = args.ip_path
    base_model_path = args.sd_path
    vae_model_path = args.vae_path
    image_encoder_path = args.image_encoder_path
    ip_ckpt = args.ip_ckpt
    inpaint_controlnet_model_path = args.controlnet_path
    device = "cuda"
    
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(ip_path+vae_model_path).to(dtype=torch.float16)
    
    # load SD Inpainting pipe
    inpaint_controlnet = ControlNetModel.from_pretrained(ip_path+inpaint_controlnet_model_path, torch_dtype=torch.float16)
    torch.cuda.empty_cache()
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        ip_path+base_model_path,
        controlnet=inpaint_controlnet, 
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to(device)
    
    # load ip-adapter
    #ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    ip_model = IPAdapterPlus(pipe, ip_path+image_encoder_path, ip_path+ip_ckpt, device, num_tokens=16)

    ############ start retargeting #################
    start_time = time.time()
    for img_name in image_name_list:
        image_name = img_name[:-4]
        print(image_name)
        img = Image.open(image_file+image_name+'.png').convert('RGB')
        W,H = (img.size)
        
        saliency = get_saliency_map(saliency_net, img, save_root=saliency_path+img_name)
        saliency_threshold = np.mean(saliency)
        saliency = np.where(saliency>saliency_threshold,saliency,0).astype(np.float32)
        
        # saliency energy to determine the important area
        saliency_energy = cal_saliency_energy(saliency, target_ratio)
        extra_energy = torch.ones([H,W]).cuda()
        extra_energy = extra_energy.cpu().numpy()
        extra_energy = extra_energy+5*(saliency_energy/255)
        extra_energy=extra_energy.astype(np.float32)
        
        ########### the maximum allowable width and height for cutting ############
        count_saliency = np.where(saliency>saliency_threshold,1,0).astype(np.float32)
        (h,w) = count_saliency.shape
        final_cut_saliency_ratio_w = cut_saliency_ratio_w
        max_cut_w = w-np.max(np.sum(count_saliency,axis=1))
        max_cut_w += int(final_cut_saliency_ratio_w*(w-max_cut_w)) 
        max_cut_h = h-np.max(np.sum(count_saliency,axis=0))
        max_cut_h += int(cut_saliency_ratio_h*(h-max_cut_h))
        
        ########## content aware seam carving ##############
        img_name = image_name+'.png'
        src = np.array(Image.open(image_file+img_name).convert('RGB'))
        src_h, src_w, _ = src.shape
        dst, to_delete, (new_h, new_w), need_expand = CSC(src, target_ratio, extra_energy, max_cut_h, max_cut_w)
        
        dst.save(args.side_out_path+args.csc_out_path+img_name)
        to_delete.save(args.side_out_path+args.mask_out_path+'seam_'+img_name)
        
        #image = np.array(Image.open(image_file+img_name).convert('RGB'))
        #mask = np.array(Image.open(args.side_out_path+args.mask_out_path+'seam_'+img_name))
        dst = np.array(dst)
        mask = np.array(to_delete)
        mask=~mask
        
        #retarget_saliency=saliency[mask].reshape((-1,new_w))
        #mask_image = dst*np.expand_dims(mask,2).repeat(3,2)
        #mask_image = Image.fromarray(mask_image)
        #mask_image.save(args.side_out_path+args.mask_out_path+img_name)
    
        if src_h/src_w > target_ratio:
            mask = mask.transpose(1,0)
        
        ############# Adaptive Repainting ####################
        dst, retarget_inpainting_mask, (target_h, target_w) = adaptive_repainting_region_determination(mask, dst, src_h, src_w, new_w, new_h, target_ratio, need_expand)
        
        control_image = make_inpaint_condition(dst, retarget_inpainting_mask)
        images = ip_model.generate(pil_image=img, num_samples=1, num_inference_steps=50,
                               seed=42, image=dst, control_image=control_image, mask_image=retarget_inpainting_mask, strength=1,
                               prompt='best quality, high quality ',
                               negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality")
        dst_np = np.array(dst)
        (h,w,c) = dst_np.shape
        pred_image = images[0].resize((w,h))
    
        if check_generated_image(pred_image, src, src_h, src_w, target_ratio, extra_energy, save_file=save_path+img_name):
            print('Generating bad case, use content_aware seam-carving only!')
            continue
            
        # Paste the visible area back onto the generated image.
        pred_image = np.array(pred_image)
        retarget_inpainting_mask=np.array(retarget_inpainting_mask).reshape((h,w,1)).repeat(3,axis=-1)
        pred_image = np.where(retarget_inpainting_mask==True,pred_image,dst)
        pred_image = Image.fromarray(pred_image)
        #pred_image = texture_edge_blurring(retarget_inpainting_mask,pred_image)
        pred_image = pred_image.resize((target_w,target_h))
        pred_image.save(save_path+img_name)
    
    end_time = time.time()
    running_time = end_time-start_time
    print('Retargeting finished! Time cost : %.5f s per image' %(running_time/len(image_name_list)))



