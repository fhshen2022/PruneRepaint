import numpy as np
from PIL import Image
import ContentAwareSeamCarving.seam_carving as carve
import torch
from torch.autograd import Variable
import scipy
import time
from torchvision import transforms
import torch.nn.functional as F
from RGB_VST.Models.ImageDepthNet import ImageDepthNet
import argparse
from diffusers import DDIMScheduler, AutoencoderKL,StableDiffusionControlNetInpaintPipeline, ControlNetModel
from IP_Adapter.ip_adapter import IPAdapter,IPAdapterPlus
import os
from collections import OrderedDict
os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser()


parser.add_argument('--target_ratio', default="4/3", type=str, help='target retargeted aspect ratio')
parser.add_argument('--save_image_path', default="./output/retargeted_results/retargetme_4_3/", type=str, help='path to save retargeted images')
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
#args = parser.parse_known_args()[0]


# super parameters
numerator, demoninator = map(int,args.target_ratio.split('/'))
target_ratio = numerator/demoninator
save_path = args.save_image_path
cut_saliency_ratio_w = 0.5 if target_ratio>4/3 else 0.3#横向最多允许丢失0.3的显著性区域；当比例过于极端时允许丢失0.5。
cut_saliency_ratio_h = 0.1 if target_ratio>4/3 else 0.05
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(args.side_out_path):
    os.mkdir(args.side_out_path)
if not os.path.exists(args.side_out_path+args.csc_out_path):
    os.mkdir(args.side_out_path+args.csc_out_path)
if not os.path.exists(args.side_out_path+args.mask_out_path):
    os.mkdir(args.side_out_path+args.mask_out_path)

#read data
image_file = args.input_image_path
saliency_path = args.save_saliency_path
image_name_list = os.listdir(image_file)


# load saliency detection model (multi-gpu)
saliency_net = ImageDepthNet(args).cuda()
model_path = args.save_model_dir + 'RGB_VST.pth'
state_dict = torch.load(model_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
saliency_net.load_state_dict(new_state_dict)
print('Salient Detection Model loaded from {}'.format(model_path))


########load AR##########
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




def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    tmp = image.astype(np.uint8)
    tmp = Image.fromarray(tmp[0][0])
    image = torch.from_numpy(image)
    return image




transform_to_sod = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])

start_time = time.time()
for img_name in image_name_list:
    image_name = img_name[:-4]
    print(image_name)
    img = Image.open(image_file+image_name+'.png').convert('RGB')
    W,H = (img.size)
    transform_back = transforms.Compose([transforms.ToPILImage(),transforms.Resize((H, W))])
    img_tensor = transform_to_sod(img).unsqueeze(0).cuda()
    img_tensor = Variable(img_tensor)
    
    output_saliency, _ = saliency_net(img_tensor)
    mask_1_16, mask_1_8, mask_1_4, saliency = output_saliency
    saliency = F.sigmoid(saliency)
    saliency_image = saliency[0].cpu().squeeze(0)
    
    saliency_image = transform_back(saliency_image)
    saliency_image.save(saliency_path+img_name)
    saliency = Image.open(saliency_path+image_name+'.png')
    saliency = np.array(saliency)
    saliency_threshold = np.mean(saliency)
    saliency = np.where(saliency>saliency_threshold,saliency,0).astype(np.float32)
    
    #saliency_energy
    saliency_energy = np.zeros(saliency.shape)
    points_h = []
    points_w = []
    for i in range(saliency.shape[0]):
        for j in range(saliency.shape[1]):
            if saliency[i][j]>0:
                points_h.append(i)
                points_w.append(j)
    center_h,center_w = sum(points_h)/len(points_h), sum(points_w)/len(points_w)
    for i in range(saliency.shape[0]):
        for j in range(saliency.shape[1]):
            if saliency[i][j]>0:
                if saliency.shape[0]/saliency.shape[1] > target_ratio:
                    saliency_energy[i][j] = saliency[i][j]*(1-abs(i-center_h)/saliency.shape[0])#spacial prior
                else:
                    saliency_energy[i][j] = saliency[i][j]*(1-abs(j-center_w)/saliency.shape[1])#spacial prior
                saliency_energy[i][j] = saliency_energy[i][j] if saliency_energy[i][j]>25.5 else 25.5
    
    # extra_energy to assist canny energy to determine the importance area
    extra_energy = torch.ones([H,W]).cuda()
    extra_energy = extra_energy.cpu().numpy()
    extra_energy = extra_energy+5*(saliency_energy/255)
    extra_energy=extra_energy.astype(np.float32)
    
    
    ###########max_cut_width############
    count_saliency = np.where(saliency>saliency_threshold,1,0).astype(np.float32)
    (h,w) = count_saliency.shape
    final_cut_saliency_ratio_w = cut_saliency_ratio_w
    max_cut_w = w-np.max(np.sum(count_saliency,axis=1))
    max_cut_w += int(final_cut_saliency_ratio_w*(w-max_cut_w)) 
    max_cut_h = h-np.max(np.sum(count_saliency,axis=0))
    max_cut_h += int(cut_saliency_ratio_h*(h-max_cut_h))
    
    ##########content aware seam carving##############
    img_name = image_name+'.png'
    src = np.array(Image.open(image_file+img_name).convert('RGB'))
    src_h, src_w, _ = src.shape
    need_expand = False
    if src_h/src_w > target_ratio:#图片高宽比过大，需要旋转后裁剪宽度
        #著物体小，可以完全裁剪
        new_h = src_w
        new_w = int(src_w*target_ratio)
        src = src.transpose(1,0,2)
        extra_energy = extra_energy.transpose(1,0)

        if src_h-new_w>max_cut_h:#显著物体过大，只能裁剪小部分显
            new_w = int(src_h-max_cut_h)
            new_h = int(new_w/target_ratio)
            need_expand = True

    else:#图片高宽比过小，直接裁剪宽度
        new_h = src_h
        new_w = int(src_h/target_ratio)
        if src_w-new_w>max_cut_w:
            new_w = int(src_w-max_cut_w)
            new_h = int(new_w*target_ratio)

    if src_h/src_w > target_ratio:
        if need_expand:
            dst,to_delete = carve.highlight_deleted_seam(
                src, 
                size = (new_w,src_w),
                energy_mode="forward" ,
                aux_energy_rate = extra_energy
            )
        else:
            dst,to_delete = carve.highlight_deleted_seam(
                src,  
                size = (new_w,new_h),
                energy_mode="forward" ,
                aux_energy_rate = extra_energy
            )

    else:
        dst,to_delete = carve.highlight_deleted_seam(
            src, 
            size = (new_w,src_h),
            energy_mode="forward" ,
            aux_energy_rate = extra_energy
        )

    if src_h/src_w > target_ratio:
        dst = Image.fromarray(dst.transpose(1,0,2))
        to_delete = Image.fromarray(to_delete.transpose(1,0))
    else:
        dst = Image.fromarray(dst)
        to_delete = Image.fromarray(to_delete)

    dst.save(args.side_out_path+args.csc_out_path+img_name)
    to_delete.save(args.side_out_path+args.mask_out_path+'seam_'+img_name)
    
    image = np.array(Image.open(image_file+img_name).convert('RGB'))
    mask = np.array(Image.open(args.side_out_path+args.mask_out_path+'seam_'+img_name))
    mask=~mask
    
    retarget_saliency=saliency[mask].reshape((-1,new_w))
    
    #mask_image = image*np.expand_dims(mask,2).repeat(3,2)
    #mask_image = Image.fromarray(mask_image)
    #mask_image.save(args.side_out_path+args.mask_out_path+img_name)
    #h,w = H,W

    if src_h/src_w > target_ratio:
        mask = mask.transpose(1,0)
        
    mask_tmp = np.where(mask>0,0,1)
    inpainting_mask = mask_tmp.copy()


    conv_kernel = [1 for i in range(25)]
    for i in range((inpainting_mask).shape[0]):
        inpainting_mask[i] = scipy.signal.convolve(mask_tmp[i],conv_kernel,mode='same')
        
    inpainting_mask = np.where(inpainting_mask>7,True,False)#周围25个像素有7个以上像素被删除，该位置需要inpainting

    if src_h/src_w > target_ratio:
        if need_expand:
            retarget_inpainting_mask = (inpainting_mask)[mask].reshape((src_w,new_w)).transpose(1,0)
        else:
            retarget_inpainting_mask = (inpainting_mask)[mask].reshape((new_h,new_w)).transpose(1,0)
    else:
        retarget_inpainting_mask = (inpainting_mask)[mask].reshape((src_h,new_w))


    if src_h/src_w > target_ratio:
        if need_expand:
            left_part = np.ones((new_w,int(new_h-retarget_inpainting_mask.shape[1])//2+5))>0
            right_part = np.ones((new_w,int(new_h-retarget_inpainting_mask.shape[1]-left_part.shape[1]+10)))>0
            retarget_inpainting_mask = np.concatenate((left_part,retarget_inpainting_mask[:,5:src_w-5],right_part),axis=1)

    else:
        if retarget_inpainting_mask.shape[0]<new_h:
            top_part = np.ones((int(new_h-retarget_inpainting_mask.shape[0])//2+5,new_w))>0
            down_part = np.ones((int(new_h-retarget_inpainting_mask.shape[0]-top_part.shape[0]+10),new_w))>0
            retarget_inpainting_mask = np.concatenate((top_part,retarget_inpainting_mask[5:src_h-5,:],down_part),axis=0)
            #new_retarget_saliency = np.concatenate((top_part,retarget_saliency[4:src_h-4,:],down_part),axis=0)
    
    
    (target_h,target_w) = retarget_inpainting_mask.shape
    inpainting_mask_image = Image.fromarray(retarget_inpainting_mask)
    dst = np.array(Image.open(args.side_out_path+args.csc_out_path+img_name))

    if src_h/src_w > target_ratio:
        if need_expand:
            expand_w = (new_h-src_w)//2
            dst_array = np.concatenate((np.uint8(255*np.ones((new_w,expand_w,3))),dst,np.uint8(255*np.ones((new_w,new_h-src_w-expand_w,3)))),axis=1)
            dst = Image.fromarray(dst_array)
        else:
            dst = Image.fromarray(dst)
    else:
        if dst.shape[0]<new_h:
            expand_h = (new_h-src_h)//2
            dst_array = np.concatenate((np.uint8(255*np.ones((expand_h,new_w,3))),dst,np.uint8(255*np.ones((new_h-src_h-expand_h,new_w,3)))),axis=0)
            dst = Image.fromarray(dst_array)
        else:
            dst = Image.fromarray(dst)

    
    retarget_inpainting_mask = Image.fromarray(retarget_inpainting_mask)
    if target_ratio<1:
        dst = dst.resize((512,int(512*target_ratio)))
        retarget_inpainting_mask = retarget_inpainting_mask.resize((512,int(512*target_ratio)))
    else:
        dst = dst.resize((int(512/target_ratio),512))
        retarget_inpainting_mask = retarget_inpainting_mask.resize((int(512/target_ratio),512))
    
    control_image = make_inpaint_condition(dst, retarget_inpainting_mask)
    image = Image.open(image_file+image_name+'.png').convert('RGB')
    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50,
                           seed=42, image=dst, control_image=control_image, mask_image=retarget_inpainting_mask, strength=1,
                           prompt='best quality, high quality ',
                           negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality")
    

    dst_np = np.array(dst)
    (h,w,c) = dst_np.shape
    pred_image = images[0].resize((w,h))

    
    if np.array(pred_image).sum()==0:
        #判断是否生成黑图，生成失败图像取消生成策略
        if src_h/src_w > target_ratio:
            new_h = src_w
            new_w = int(src_w*target_ratio)
            src = src.transpose(1,0,2)
            extra_energy = extra_energy.transpose(1,0)
            dst,to_delete = carve.highlight_deleted_seam(
                src, 
                size = (new_h,new_w),
                energy_mode="forward" ,
                aux_energy_rate = extra_energy
             )
        else:
            new_h = src_h
            new_w = int(src_h/target_ratio)
            dst,to_delete = carve.highlight_deleted_seam(
                src, 
                size = (new_w,new_h),
                energy_mode="forward" ,
                aux_energy_rate = extra_energy
             )
        
        if src_h/src_w > target_ratio:
            dst = Image.fromarray(dst)
            dst.save(save_path+img_name)
        else:
            dst = Image.fromarray(dst)
            dst.save(save_path+img_name) 
        continue
        
    
    #贴原图
    pred_image = np.array(pred_image)
    retarget_inpainting_mask=np.array(retarget_inpainting_mask).reshape((h,w,1)).repeat(3,axis=-1)
    pred_image = np.where(retarget_inpainting_mask==True,pred_image,dst)
    pred_image = Image.fromarray(pred_image)
    
    '''
    #找出贴图边界并模糊处理
    retarget_inpainting_mask = retarget_inpainting_mask[:,:,1]
    kernel = np.ones((5,5))/25
    retarget_inpainting_mask = np.where(retarget_inpainting_mask==True ,255,0)
    retarget_inpainting_mask = scipy.signal.convolve2d(retarget_inpainting_mask,np.ones((3,3))/9,'same',fillvalue=255)
    fuse_inpainting_mask = np.where(retarget_inpainting_mask>254,0,retarget_inpainting_mask)
    fuse_inpainting_mask = np.where(fuse_inpainting_mask>0,True,False)
    pred_img_array = np.array(pred_image)
    blur_image = pred_img_array
    for i in range(pred_img_array.shape[-1]):
        blur_image[:,:,i] = scipy.signal.convolve2d(pred_img_array[:,:,i],kernel,'same',fillvalue=0)
    fuse_inpainting_mask = np.expand_dims(fuse_inpainting_mask,2).repeat(3,2)
    pred_image = np.where(fuse_inpainting_mask==True,blur_image, pred_image)
    pred_image = Image.fromarray(pred_image)
    '''
    
    pred_image = pred_image.resize((target_w,target_h))
    pred_image.save(save_path+img_name)


end_time = time.time()
running_time = end_time-start_time
print('Retargeting finished! Time cost : %.5f s per image' %(running_time/len(image_name_list)))



