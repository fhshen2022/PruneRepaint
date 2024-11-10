import numpy as np
from PIL import Image
import torch
import ContentAwareSeamCarving.seam_carving as carve
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import scipy

transform_to_sod = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
def get_saliency_map(saliency_net,img,save_root):
    W,H = (img.size)
    transform_back = transforms.Compose([transforms.ToPILImage(),transforms.Resize((H, W))])
    img_tensor = transform_to_sod(img).unsqueeze(0).cuda()
    img_tensor = Variable(img_tensor)
    
    output_saliency, _ = saliency_net(img_tensor)
    mask_1_16, mask_1_8, mask_1_4, saliency = output_saliency
    saliency = F.sigmoid(saliency)
    saliency_image = saliency[0].cpu().squeeze(0)
    
    saliency_image = transform_back(saliency_image)
    saliency_image.save(save_root)
    saliency = np.array(saliency_image)
    #saliency_threshold = np.mean(saliency)
    #saliency = np.where(saliency>saliency_threshold,saliency,0).astype(np.float32)
    return saliency


def cal_saliency_energy(saliency, target_ratio):
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
    return saliency_energy


def CSC(src,target_ratio,extra_energy,max_cut_h,max_cut_w):
    #content_aware seam-carving
    
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
    
    return dst, to_delete, (new_h, new_w),need_expand



def adaptive_repainting_region_determination(mask, dst, src_h, src_w, new_w, new_h, target_ratio, need_expand):
    mask_tmp = np.where(mask>0,0,1)
    inpainting_mask = mask_tmp.copy()

    #滑动窗口长度为25
    conv_kernel = [1 for i in range(25)]
    for i in range((inpainting_mask).shape[0]):
        inpainting_mask[i] = scipy.signal.convolve(mask_tmp[i],conv_kernel,mode='same')
    inpainting_mask = np.where(inpainting_mask>7,True,False)#滑动窗口中25个像素有7个以上像素被删除，该位置需要inpainting

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

    return dst, retarget_inpainting_mask,(target_h,target_w)


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

def check_generated_image(pred_image,src,src_h,src_w,target_ratio, extra_energy,save_file):
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
        dst = Image.fromarray(dst)
        dst.save(save_file)
        return False
    else:
        return False

def texture_edge_blurring(retarget_inpainting_mask,pred_image):
    #Detect and blur the edges of the texture.
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
    return pred_image
