"""Attention under the two image preparations, same checkpoint and same frame.

s3_applied.npz            GR00T_SQUARE_CROP=1 with roi_hybrid.json -- the applied
                         configuration, and the one the checkpoint was trained under
s3_letterbox.npz  GR00T_SQUARE_CROP=0 -- every view padded to square

Both from attn_map.py --dump, checkpoint-30000, episode 1 frame 898. Per panel
the 64 token values are min-max normalised, which is the usual convention for
these maps: colour is comparable within a panel, not between panels.
"""
import numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"font.family":"DejaVu Sans","font.size":8,"savefig.dpi":220})
SIDE=256

D={"applied": np.load("sweepattn/s1_applied.npz",allow_pickle=True),
   "letterbox": np.load("sweepattn/s1_letterbox.npz",allow_pickle=True)}
cams=[str(c) for c in D["applied"]["cams"]]
assert cams==[str(c) for c in D["letterbox"]["cams"]], "camera order differs between dumps"

#the point of the comparison is that only the image preparation differs, so the
#frame and the language prompt must be identical in both runs
INSTR=str(D["applied"]["instruction"])
FRAME=int(D["applied"]["frame"])
assert INSTR==str(D["letterbox"]["instruction"]), (
    f"different instruction: {INSTR!r} vs {str(D['letterbox']['instruction'])!r}")
assert FRAME==int(D["letterbox"]["frame"]), "different frame"
print(f"both runs: frame {FRAME}, instruction {INSTR!r}")

def panels(d,cam):
    hm=d[f"heat_{cam}"].astype(np.float32); img=d[f"img_{cam}"]
    lo,hi=float(hm.min()),float(hm.max())
    nrm=(hm-lo)/max(hi-lo,1e-12)
    up=cv2.resize(nrm,(SIDE,SIDE),interpolation=cv2.INTER_LINEAR)
    up=cv2.GaussianBlur(up,(0,0),SIDE/hm.shape[0]/2.0)
    up=np.clip((up-up.min())/max(up.max()-up.min(),1e-12),0,1)
    jet=cv2.cvtColor(cv2.applyColorMap((up*255).astype(np.uint8),cv2.COLORMAP_JET),
                     cv2.COLOR_BGR2RGB)
    base=cv2.resize(img,(SIDE,SIDE),interpolation=cv2.INTER_AREA)
    return base, cv2.addWeighted(base,0.5,jet,0.5,0)

ROWS=[("applied","base","applied crop\ninput"),
      ("applied","ov",  "applied crop\nattention"),
      ("letterbox","base","uniform letterbox\ninput"),
      ("letterbox","ov",  "uniform letterbox\nattention")]

fig,ax=plt.subplots(4,3,figsize=(6.3,8.9),
                    gridspec_kw={"wspace":0.04,"hspace":0.05,"top":0.93})
#the prompt sits above the column titles, clear of every panel
fig.text(0.5,0.982,f"instruction: \u201c{INSTR}\u201d",
         ha="center",va="bottom",fontsize=8.5,style="italic")
for r,(key,kind,lbl) in enumerate(ROWS):
    for c,cam in enumerate(cams):
        base,ov=panels(D[key],cam)
        ax[r,c].imshow(base if kind=="base" else ov)
        ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
        for s in ax[r,c].spines.values(): s.set_linewidth(0.5)
        if r==0: ax[r,c].set_title(cam,fontsize=8.5,pad=4)
    ax[r,0].set_ylabel(lbl,fontsize=8,labelpad=6)
fig.savefig("figs/attn_map.png",bbox_inches="tight",facecolor="white")
plt.close(fig)
print("wrote figs/attn_map.png")
for k,d in D.items():
    u=float(d["uniform"])
    rr=np.concatenate([(d[f"heat_{c}"]/u).ravel() for c in cams])
    print(f"  {k:10s} token range {rr.min():.2f}x .. {rr.max():.2f}x fair share")
