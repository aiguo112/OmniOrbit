#!/usr/bin/env python3
"""Full test-split classification eval -> per_class_acc.csv, confusion, confidences."""
import os, json, csv
import numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT="space_erp_final"; CKPT="runs/cls_r50/best.pt"; OUT="classify_testset"
IMG_H,IMG_W=384,768
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MEAN,STD=[0.485,0.456,0.406],[0.229,0.224,0.225]
os.makedirs(OUT,exist_ok=True)

ck=torch.load(CKPT,map_location=DEVICE)
classes=ck["classes"]; arch=ck["arch"]; n=len(classes)
idx={c:i for i,c in enumerate(classes)}
short=[c.split("_",1)[1] if "_" in c else c for c in classes]

m=getattr(torchvision.models,arch)(weights=None)
m.fc=nn.Linear(m.fc.in_features,n)
m.load_state_dict(ck["model"]); m.to(DEVICE).eval()

test=json.load(open(os.path.join(ROOT,"splits.json")))["splits"]["test"]
tf=T.Compose([T.Resize((IMG_H,IMG_W)),T.ToTensor(),T.Normalize(MEAN,STD)])

class DS(Dataset):
    def __init__(self,stems): self.stems=stems
    def __len__(self): return len(self.stems)
    def __getitem__(self,i):
        stem=self.stems[i]; cls=stem.split("/")[0]
        img=tf(Image.open(os.path.join(ROOT,"image",stem+".png")).convert("RGB"))
        return img, idx.get(cls,-1), stem

# keep only stems whose class is known
test=[s for s in test if s.split("/")[0] in idx]
dl=DataLoader(DS(test),batch_size=16,num_workers=8,shuffle=False)
print(f"{len(test)} test frames, {n} classes, arch={arch}")

cm=np.zeros((n,n),int); rows=[]; correct=0; total=0
sm=nn.Softmax(dim=1)
with torch.no_grad():
    for imgs,labs,stems in dl:
        imgs=imgs.to(DEVICE)
        p=sm(m(imgs)); conf,pred=p.max(1)
        pred=pred.cpu().numpy(); conf=conf.cpu().numpy(); labs=labs.numpy()
        for st,tr,pr,cf in zip(stems,labs,pred,conf):
            if tr<0: continue
            cm[tr,pr]+=1; correct+=int(pr==tr); total+=1
            rows.append((st,classes[tr],classes[pr],int(pr==tr),f"{cf:.4f}"))

with open(os.path.join(OUT,"predictions.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["stem","true","pred","correct","confidence"]); w.writerows(rows)

# per-class accuracy
per=[]
for i,c in enumerate(classes):
    tot=cm[i].sum(); acc=cm[i,i]/tot if tot else 0.0
    per.append((c,int(tot),int(cm[i,i]),f"{acc:.4f}"))
with open(os.path.join(OUT,"per_class_acc.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["class","n","correct","accuracy"]); w.writerows(per)
print(f"overall top-1 = {correct/max(total,1):.4f} ({correct}/{total})")
for c,tot,cor,acc in per: print(f"  {c:28s} {acc}  ({cor}/{tot})")

# row-normalized confusion matrix
cmn=cm/np.clip(cm.sum(1,keepdims=True),1,None)
fig,ax=plt.subplots(figsize=(9,8))
im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(short,rotation=45,ha="right",fontsize=9); ax.set_yticklabels(short,fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(n):
    for j in range(n):
        v=cmn[i,j]
        if v>=0.01: ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                            color="white" if v>0.5 else "black",fontsize=7)
fig.colorbar(im,fraction=0.046,pad=0.04); fig.tight_layout()
fig.savefig(os.path.join(OUT,"confusion_matrix.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"confusion_matrix.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote ->",OUT)
