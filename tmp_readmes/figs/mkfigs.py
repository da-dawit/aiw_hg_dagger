"""Figures for Section 6 of the manual, from real recorded frames.

Nothing here is illustrative: the frames come from the reference session and
the motion-energy profile is computed from the head camera's own video.
"""
import glob, json, subprocess, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.family":"DejaVu Sans","font.size":8,
                 "axes.linewidth":0.6,"xtick.major.width":0.6,
                 "ytick.major.width":0.6,"savefig.dpi":200})
HUMAN=(170/255,45/255,45/255); AUTO=(30/255,95/255,175/255)

D="/home/robotis/robot_aiworker/datasets/dagger15/videos/chunk-000"
CAM={"cam_left_head":"observation.images.rgb.cam_left_head",
     "cam_left_wrist":"observation.images.rgb.cam_left_wrist",
     "cam_right_wrist":"observation.images.rgb.cam_right_wrist",
     "cam_right_head":"observation.images.rgb.cam_right_head"}

def frames(cam, ep=0, idxs=(1800,)):
    """Decode specific frame indices from one episode's video."""
    path=f"{D}/{CAM[cam]}/episode_{ep:06d}.mp4"
    cap=cv2.VideoCapture(path); out={}
    for i in sorted(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok,fr=cap.read()
        if ok: out[i]=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release(); return out

def seq(cam, ep=0, start=1700, n=120, stride=3):
    path=f"{D}/{CAM[cam]}/episode_{ep:06d}.mp4"
    cap=cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    out=[]
    for k in range(n*stride):
        ok,fr=cap.read()
        if not ok: break
        if k%stride==0: out.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release(); return out

# ---------------------------------------------------------------- fig 1
f=frames("cam_left_head")[1800]; print("head frame", f.shape)
fig,ax=plt.subplots(2,2,figsize=(7.2,4.4))
order=[("cam_left_head","cam_left_head","ZED left  672x376"),
       ("cam_right_head","cam_right_head","ZED right  672x376"),
       ("cam_left_wrist","cam_left_wrist","D405 left wrist  240x424"),
       ("cam_right_wrist","cam_right_wrist","D405 right wrist  240x424")]
for a,(key,name,cap) in zip(ax.ravel(), order):
    im=frames(key)[1800]
    a.imshow(im); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{name}\n{cap}", fontsize=7.5, pad=3)
    for s in a.spines.values(): s.set_linewidth(0.6)
fig.tight_layout(); fig.savefig("figs/camera_views.png", bbox_inches="tight"); plt.close(fig)
print("wrote camera_views.png")
