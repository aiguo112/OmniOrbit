#!/usr/bin/env python3
"""Classify candidates_gt/ with cls_r50 -> classify_candidates.csv + confusion matrix + grid."""
import os, glob, csv, json
import numpy as np, torch, torch.nn as nn, torchvision
from PIL import Image
from torchvision import transforms as T
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT="candidates_gt"; CKPT="runs/cls_r50/best.pt"; OUT="classify_eval"
IMG_H,IMG_W=384,768
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MEAN,STD=[0.485,0.456,0.406],[0.229,0.224,0.225]
os.makedirs(OUT,exist_ok=True)

ck=torch.load(CKPT,map_location=DEVICE)
classes=ck["classes"]; arch=ck["arch"]; n=len(classes)
idx={c:i for i,c in enumerate(classes)}
short=[c.split("_",1)[1] if "_" in c else c for c in classes]   # strip NN_ for labels

m=getattr(torchvision.models,arch)(weights=None)
m.fc=nn.Linear(m.fc.in_features,n)
m.load_state_dict(ck["model"]); m.to(DEVICE).eval()

tf=T.Compose([T.Resize((IMG_H,IMG_W)),T.ToTensor(),T.Normalize(MEAN,STD)])
frames=sorted(glob.glob(os.path.join(GT,"*","*")))
print(f"{len(frames)} frames, {n} classes, arch={arch}")

cm=np.zeros((n,n),int); rows=[]; correct=0
with torch.no_grad():
    for fd in frames:
        cls=os.path.basename(os.path.dirname(fd)); stem=os.path.basename(fd)
        if cls not in idx: continue
        t=tf(Image.open(os.path.join(fd,"image.png")).convert("RGB")).unsqueeze(0).to(DEVICE)
        pr=int(m(t)[0].argmax()); tr=idx[cls]
        cm[tr,pr]+=1; correct+=(pr==tr)
        rows.append((cls,stem,classes[tr],classes[pr],int(pr==tr)))

with open(os.path.join("outputs", "classify_candidates.csv"), "w", newline="") as f:
    w=csv.writer(f); w.writerow(["cls","stem","true","pred","correct"]); w.writerows(rows)
acc=correct/max(len(rows),1); print(f"candidate top-1 = {acc:.4f} ({correct}/{len(rows)})")

# --- confusion matrix (row-normalized) ---
cmn=cm/np.clip(cm.sum(1,keepdims=True),1,None)
fig,ax=plt.subplots(figsize=(9,8))
im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(short,rotation=45,ha="right",fontsize=9)
ax.set_yticklabels(short,fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(n):
    for j in range(n):
        v=cmn[i,j]
        if v>=0.01: ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                            color="white" if v>0.5 else "black",fontsize=7)
fig.colorbar(im,fraction=0.046,pad=0.04)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"confusion_matrix.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"confusion_matrix.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote confusion_matrix")

# --- labeled grid: 2 per class = 26 tiles, prefer 1 correct + 1 error where available ---
by={c:[] for c in classes}
for cls,stem,tr,pr,ok in rows: by[cls].append((stem,tr,pr,ok))
pick=[]
for c in classes:
    items=by[c]; errs=[x for x in items if not x[3]]; oks=[x for x in items if x[3]]
    chosen=[]
    if oks: chosen.append(oks[0])
    if errs: chosen.append(errs[0])
    while len(chosen)<2 and len(items)>len(chosen): chosen.append(items[len(chosen)])
    for stem,tr,pr,ok in chosen[:2]: pick.append((c,stem,tr,pr,ok))

cols=2; rowsN=int(np.ceil(len(pick)/cols))
fig,axs=plt.subplots(rowsN,cols,figsize=(cols*3.2,rowsN*1.9))
for a in axs.flat: a.axis("off")
for k,(c,stem,tr,pr,ok) in enumerate(pick):
    a=axs.flat[k]
    a.imshow(Image.open(os.path.join(GT,c,stem,"image.png")).convert("RGB"))
    ts=tr.split("_",1)[-1]; ps=pr.split("_",1)[-1]
    a.set_title(f"T:{ts}\nP:{ps}",fontsize=7,color=("green" if ok else "red"),pad=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"labeled_grid.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"labeled_grid.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote labeled_grid ->",OUT)
