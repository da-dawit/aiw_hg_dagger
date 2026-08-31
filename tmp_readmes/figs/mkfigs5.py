"""The three image-preparation figures, with no text beyond axis and panel labels.

Numbers live in the manual's tables, not baked into the pictures.
Source: episode 0 of datasets/dagger15, every fourth frame.
"""
import numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":0.6,
                 "xtick.major.width":0.6,"ytick.major.width":0.6,"savefig.dpi":220})
HUM=(170/255,45/255,45/255); AUT=(30/255,95/255,175/255)
D="/home/robotis/robot_aiworker/datasets/dagger15/videos/chunk-000"
P={k:f"{D}/observation.images.rgb.{k}/episode_000000.mp4" for k in
   ("cam_left_head","cam_left_wrist","cam_right_wrist")}

def grab(cam,i):
    c=cv2.VideoCapture(P[cam]); c.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=c.read(); c.release()
    return cv2.cvtColor(f,cv2.COLOR_BGR2RGB)

def col_motion(cam,stride=4):
    c=cv2.VideoCapture(P[cam]); prev=None; acc=None; k=0
    while True:
        ok,f=c.read()
        if not ok: break
        if k%stride==0:
            g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                d=np.abs(g-prev).sum(axis=0); acc=d if acc is None else acc+d
            prev=g
        k+=1
    c.release(); return acc

head=grab("cam_left_head",1800); H,W=head.shape[:2]
lo=(W-H)//2; hi=lo+H
m=col_motion("cam_left_head")
wr=grab("cam_right_wrist",1800); wh,ww=wr.shape[:2]; cut=wh-ww

# ---------------- methods ----------------
fig,ax=plt.subplots(1,3,figsize=(7.4,2.2))
ax[0].imshow(head); ax[0].set_title("recorded, 672$\\times$376",fontsize=8)
lb=np.zeros((W,W,3),np.uint8); off=(W-H)//2; lb[off:off+H]=head
ax[1].imshow(lb); ax[1].set_title("letterbox, 672$\\times$672",fontsize=8)
ax[1].add_patch(patches.Rectangle((0,0),W,off,fc="none",ec=HUM,lw=0.8,hatch="///"))
ax[1].add_patch(patches.Rectangle((0,off+H),W,off,fc="none",ec=HUM,lw=0.8,hatch="///"))
ax[2].imshow(head); ax[2].set_title("square crop, 376$\\times$376",fontsize=8)
ax[2].add_patch(patches.Rectangle((0,0),lo,H,fc="k",alpha=0.6,ec="none"))
ax[2].add_patch(patches.Rectangle((hi,0),W-hi,H,fc="k",alpha=0.6,ec="none"))
ax[2].add_patch(patches.Rectangle((lo,0),H,H,fc="none",ec=AUT,lw=1.0))
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig("figs/crop_methods.png",bbox_inches="tight",facecolor="white")
plt.close(fig)

# ---------------- evidence ----------------
fig=plt.figure(figsize=(7.4,3.3))
gs=fig.add_gridspec(2,2,width_ratios=[2.0,1.0],height_ratios=[1.35,1.0],hspace=0.30,wspace=0.20)
a0=fig.add_subplot(gs[0,0]); a0.imshow(head); a0.set_xticks([]); a0.set_yticks([])
a0.axvline(lo,color=AUT,lw=1.0); a0.axvline(hi,color=AUT,lw=1.0)
a0.add_patch(patches.Rectangle((0,0),lo,H,fc=HUM,alpha=0.22,ec="none"))
a0.add_patch(patches.Rectangle((hi,0),W-hi,H,fc=HUM,alpha=0.22,ec="none"))
a0.set_title("cam_left_head",fontsize=8)
a1=fig.add_subplot(gs[1,0]); x=np.arange(W)
a1.fill_between(x[:lo],m[:lo],color=HUM,alpha=0.35,lw=0)
a1.fill_between(x[hi:],m[hi:],color=HUM,alpha=0.35,lw=0)
a1.fill_between(x[lo:hi],m[lo:hi],color=AUT,alpha=0.25,lw=0)
a1.plot(x,m,color="k",lw=0.5)
a1.axvline(lo,color=AUT,lw=1.0); a1.axvline(hi,color=AUT,lw=1.0)
a1.set_xlim(0,W); a1.set_yticks([])
a1.set_xlabel("image column",fontsize=8); a1.set_ylabel("motion energy",fontsize=8)
a2=fig.add_subplot(gs[:,1]); a2.imshow(wr); a2.set_xticks([]); a2.set_yticks([])
a2.add_patch(patches.Rectangle((0,0),ww,cut,fc=HUM,alpha=0.28,ec="none"))
a2.axhline(cut,color=AUT,lw=1.0)
a2.set_title("cam_right_wrist",fontsize=8)
fig.savefig("figs/crop_evidence.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

# ---------------- applied ----------------
def prep_head(img):
    m_=max(img.shape[:2]); c=np.zeros((m_,m_,3),np.uint8)
    o=(m_-img.shape[0])//2; c[o:o+img.shape[0]]=img
    return cv2.resize(c,(256,256),interpolation=cv2.INTER_AREA)
def prep_wrist(img):
    h,w=img.shape[:2]; return cv2.resize(img[h-w:h],(256,256),interpolation=cv2.INTER_AREA)
ins=[("cam_left_head",prep_head(head)),
     ("cam_left_wrist",prep_wrist(grab("cam_left_wrist",1800))),
     ("cam_right_wrist",prep_wrist(wr))]
fig,ax=plt.subplots(1,3,figsize=(7.4,2.65))
for a,(nm,im) in zip(ax,ins):
    a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_title(nm,fontsize=8,pad=3)
fig.tight_layout(); fig.savefig("figs/crop_applied.png",bbox_inches="tight",facecolor="white")
plt.close(fig)
print("regenerated crop_methods, crop_evidence, crop_applied -- no text baked in")
