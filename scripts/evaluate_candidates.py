#!/usr/bin/env python3
"""Per-frame evaluation over candidates_gt/ -> eval_candidates/ + eval_per_frame.csv."""
import os, glob, csv
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np, torch, cv2
from PIL import Image
import segmentation_models_pytorch as smp

GT   = "candidates_gt"
OUT  = "eval_candidates"
CSV  = os.path.join("outputs", "eval_per_frame.csv")
IMG_W, IMG_H = 1024, 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
BG_THRESH = 1e6   # depth >= this is empty space

PALETTE = {0:(0,0,0),1:(220,50,50),2:(50,200,80),3:(50,90,220),4:(220,200,50),
  5:(200,100,30),6:(160,60,200),7:(60,200,200),8:(240,130,200),9:(250,160,40),
  10:(120,80,220),11:(30,120,255),12:(190,190,190),13:(120,100,80)}
P_ARR = np.array([PALETTE[i] for i in range(14)], np.uint8)          # for saving PNGs
_P64 = P_ARR.astype(np.int64)
P_PACK = (_P64[:,0]<<16)|(_P64[:,1]<<8)|_P64[:,2]                     # int64 packing, no overflow
ARCH = {"unet":smp.Unet,"unetpp":smp.UnetPlusPlus,"deeplabv3plus":smp.DeepLabV3Plus,
        "fpn":smp.FPN,"pspnet":smp.PSPNet}
SEG_MODELS   = [("fpn","resnet50","runs/seg_fpn_r50"),("unet","efficientnet-b0","runs/seg_unet_effb0")]
DEPTH_MODELS = [("fpn","resnet34","runs/depth_fpn_r34"),("deeplabv3plus","resnet34","runs/depth_dlv3p_r34"),
                ("pspnet","resnet34","runs/depth_pspnet_r34")]

def load(arch,enc,ckpt,classes):
    m=ARCH[arch](encoder_name=enc,encoder_weights=None,in_channels=3,classes=classes).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(ckpt,"best.pt"),map_location=DEVICE)["model"]); m.eval(); return m

def prep(path):
    im=Image.open(path).convert("RGB").resize((IMG_W,IMG_H),Image.BILINEAR)
    arr=np.asarray(im,np.float32)/255.0
    return im, torch.from_numpy(((arr-MEAN)/STD).transpose(2,0,1)[None]).to(DEVICE)

def mask_to_label(path):
    rgb=np.asarray(Image.open(path).convert("RGB").resize((IMG_W,IMG_H),Image.NEAREST),np.int64)
    packed=(rgb[...,0]<<16)|(rgb[...,1]<<8)|rgb[...,2]
    lab=np.zeros(packed.shape,np.uint8)
    for i in range(14): lab[packed==P_PACK[i]]=i
    return lab

def read_depth_gt(path):
    d=cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d.ndim==3: d=d[...,0]
    return d.astype(np.float32)

def fg_iou(pred,gt):
    ious=[]
    for c in range(1,14):
        p,g=(pred==c),(gt==c)
        u=(p|g).sum()
        if g.sum()>0: ious.append((p&g).sum()/u if u>0 else 0.0)
    return float(np.mean(ious)) if ious else float("nan")

def depth_metrics(pred,gt):
    m=(gt>0)&(gt<BG_THRESH)&np.isfinite(gt)
    if m.sum()<50: return {}
    p=pred[m].astype(np.float64); g=gt[m].astype(np.float64)
    p=np.clip(p,1e-6,None)
    s=np.exp(np.median(np.log(g)-np.log(p)))   # scale-invariant log alignment
    p=p*s
    absrel=np.mean(np.abs(p-g)/g); rmse=np.sqrt(np.mean((p-g)**2))
    r=np.maximum(p/g,g/p); d1=np.mean(r<1.25)
    return {"absrel":float(absrel),"rmse":float(rmse),"d1":float(d1)}

def cdepth(d,m=None):
    x=d.copy().astype(np.float32)
    if m is None: m=np.isfinite(x)
    lo,hi=np.percentile(x[m],2),np.percentile(x[m],98)
    xn=np.clip((x-lo)/max(hi-lo,1e-6),0,1)
    return cv2.applyColorMap((xn*255).astype(np.uint8),cv2.COLORMAP_INFERNO)[...,::-1]

segs=[(k,load(k,e,c,14)) for k,e,c in SEG_MODELS]
depths=[(k,load(k,e,c,1)) for k,e,c in DEPTH_MODELS]
frames=sorted(glob.glob(os.path.join(GT,"*","*")))
print(f"{len(frames)} frames, {len(segs)} seg + {len(depths)} depth models")

new=not os.path.exists(CSV)
cf=open(CSV,"a",newline=""); w=csv.writer(cf)
if new: w.writerow(["cls","stem","task","model","metric1_name","metric1","metric2_name","metric2","metric3_name","metric3"])

for i,fd in enumerate(frames):
    cls=os.path.basename(os.path.dirname(fd)); stem=os.path.basename(fd)
    od=os.path.join(OUT,cls,stem)
    if os.path.isdir(od): continue          # resume: skip done
    os.makedirs(od,exist_ok=True)
    rgb,t=prep(os.path.join(fd,"image.png")); rgb.save(os.path.join(od,"rgb.png"))
    gt_lab=mask_to_label(os.path.join(fd,"mask.png"))
    Image.fromarray(P_ARR[gt_lab]).save(os.path.join(od,"gt_mask.png"))
    gt_d=read_depth_gt(os.path.join(fd,"depth.exr"))
    dm=(gt_d>0)&(gt_d<BG_THRESH)&np.isfinite(gt_d)
    Image.fromarray(cdepth(gt_d,dm)).save(os.path.join(od,"gt_depth.png"))
    with torch.no_grad():
        for k,m in segs:
            pr=m(t)[0].argmax(0).cpu().numpy().astype(np.uint8)
            Image.fromarray(P_ARR[pr]).save(os.path.join(od,f"{k}_mask.png"))
            iou=fg_iou(pr,gt_lab)
            w.writerow([cls,stem,"seg",k,"fg_iou",f"{iou:.4f}","","","",""])
        for k,m in depths:
            pr=m(t)[0,0].cpu().numpy()
            Image.fromarray(cdepth(pr)).save(os.path.join(od,f"{k}_depth.png"))
            dmet=depth_metrics(pr,gt_d)
            if dmet: w.writerow([cls,stem,"depth",k,"absrel",f"{dmet['absrel']:.4f}","rmse",f"{dmet['rmse']:.3f}","d1",f"{dmet['d1']:.4f}"])
    cf.flush()
    if (i+1)%25==0: print(f"  {i+1}/{len(frames)}")
cf.close(); print("done ->",OUT,"and",CSV)
