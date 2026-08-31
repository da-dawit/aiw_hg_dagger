"""Manual figures. No captions, no explanatory text baked into any image.

Sources, all measured:
  attn_dump.npz     attn_map.py --dump, checkpoint-30000, episode 1 frame 898
  preflight.txt     06_train_dagger.sh preflight, run on this machine
  verify_human.txt  verify_dataset.py --root datasets/dagger_human --isaac
  parquet_cols.txt  pandas over datasets/dagger15
  mixture counts    meta/info.json of the three datasets
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm, colors
from matplotlib.colorbar import ColorbarBase

rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":0.6,
                 "xtick.major.width":0.6,"ytick.major.width":0.6,"savefig.dpi":220})
HUM=(170/255,45/255,45/255); AUT=(30/255,95/255,175/255); GRY=(0.55,0.55,0.55)

# ===================================================== terminal renderer
def terminal(txt, path, cols=100, fs=6.4):
    """Monospace text on a plain panel. Nothing else.

    Long lines are soft-wrapped at `cols` with a hanging indent, so one
    overlong warning does not stretch the whole panel out of legibility.
    """
    import textwrap
    lines=[]
    for l in txt.rstrip("\n").split("\n"):
        if len(l)<=cols:
            lines.append(l); continue
        ind=" "*(len(l)-len(l.lstrip()))+"  "
        lines.extend(textwrap.wrap(l,cols,subsequent_indent=ind,
                                   break_long_words=False,break_on_hyphens=False))
    w=max(len(l) for l in lines)
    fig_w=0.0655*w*(fs/6.4)*0.98
    fig_h=0.126*len(lines)*(fs/6.4)
    fig=plt.figure(figsize=(fig_w,fig_h))
    ax=fig.add_axes([0,0,1,1]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor((0.98,0.98,0.98))
    for s in ax.spines.values(): s.set_color((0.6,0.6,0.6)); s.set_linewidth(0.7)
    ax.text(0.012,0.985,"\n".join(lines),family="monospace",fontsize=fs,
            va="top",ha="left",linespacing=1.42,transform=ax.transAxes)
    fig.savefig(path,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print("wrote",path)

# ===================================================== A. attention
d=np.load("attn_dump.npz",allow_pickle=True)
u=float(d["uniform"]); cams=[str(c) for c in d["cams"]]
ratios=np.concatenate([ (d[f"heat_{c}"]/u).ravel() for c in cams ])
#symmetric in log2, so the fair share sits exactly at the centre of the scale and
#a token at 2x and one at 0.5x are equally far from it
vm=float(np.abs(np.log2(ratios)).max())
norm=colors.Normalize(-vm,vm)
cmap=cm.get_cmap("RdBu_r")

fig=plt.figure(figsize=(7.4,2.85))
gs=fig.add_gridspec(2,3,height_ratios=[1,0.042],hspace=0.16,wspace=0.05)
for i,c in enumerate(cams):
    ax=fig.add_subplot(gs[0,i])
    img=d[f"img_{c}"]; hm=np.log2(d[f"heat_{c}"]/u)
    ax.imshow(img)
    #nearest, so one token stays one square: no interpolated blobs
    ax.imshow(hm,cmap=cmap,norm=norm,alpha=0.5,interpolation="nearest",
              extent=(0,img.shape[1],img.shape[0],0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.6)
    ax.set_title(c,fontsize=8,pad=4)
cax=fig.add_subplot(gs[1,1])
ColorbarBase(cax,cmap=cmap,norm=norm,orientation="horizontal")
tk=[-1,0,1]
cax.set_xticks(tk); cax.set_xticklabels([r"$0.5\times$",r"$1\times$",r"$2\times$"],fontsize=7.5)
cax.tick_params(length=2)
fig.savefig("figs/attn_map.png",bbox_inches="tight",facecolor="white"); plt.close(fig)
print("wrote figs/attn_map.png   log2 range +/-%.2f  (%.2fx to %.2fx)"%(vm,2**-vm,2**vm))

# ===================================================== B. mixture
names=["original","dagger_human","dagger_auto"]
frames=np.array([55701,6765,31804],float)
ratio=np.array([0.60,0.30,0.10])
bysize=frames/frames.sum()
y=np.arange(3)[::-1]; h=0.34
fig,ax=plt.subplots(figsize=(5.4,1.95))
ax.barh(y+h/2,bysize*100,h,color=GRY,label="weighted by size")
ax.barh(y-h/2,ratio*100,h,color=AUT,label="applied ratio")
for i,(a,b) in enumerate(zip(bysize*100,ratio*100)):
    ax.text(a+1.2,y[i]+h/2,f"{a:.1f}%",va="center",fontsize=7)
    ax.text(b+1.2,y[i]-h/2,f"{b:.0f}%",va="center",fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([f"{n}\n{int(f):,} frames" for n,f in zip(names,frames)],fontsize=7.5)
ax.set_xlabel("share of the training mixture (%)",fontsize=8)
ax.set_xlim(0,72)
ax.legend(fontsize=7,frameon=False,loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig("figs/mixture_proportions.png",bbox_inches="tight",facecolor="white"); plt.close(fig)
print("wrote figs/mixture_proportions.png  by-size human share %.1f%%"%(bysize[1]*100))

# ===================================================== C-E. terminals
terminal(open("preflight.txt").read(),"figs/launcher_preflight.png")
terminal(open("verify_human.txt").read(),"figs/verify_output.png")
terminal(open("parquet_cols.txt").read(),"figs/parquet_columns.png")

# ===================================================== F. checkpoint sweep
import json
sw=json.load(open("sweep_local.json"))
PHASE={"0":"Grab bolt (L)","1":"Place in hole (L)","2":"Grab driver (R)",
       "3":"Screw in (R)","4":"Home"}
steps=sorted(map(int,sw))
fig,ax=plt.subplots(figsize=(5.6,2.6))
mk=["o","s","^","D","v"]
for i,(pk,pn) in enumerate(PHASE.items()):
    ys=[sw[str(s)][pk]["mean_err_mm"] for s in steps]
    ax.plot(steps,ys,marker=mk[i],ms=4,lw=1.0,label=pn)
ax.set_xscale("log")
ax.set_xticks(steps); ax.set_xticklabels([f"{s:,}" for s in steps],fontsize=7.5)
ax.set_xlabel("training step",fontsize=8)
ax.set_ylabel("held-out error (mm)",fontsize=8)
ax.set_ylim(0,34)
ax.legend(fontsize=6.8,frameon=False,ncol=2,loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y",lw=0.4,color=(0.88,0.88,0.88))
ax.set_axisbelow(True)
fig.savefig("figs/checkpoint_sweep.png",bbox_inches="tight",facecolor="white"); plt.close(fig)
print("wrote figs/checkpoint_sweep.png")
