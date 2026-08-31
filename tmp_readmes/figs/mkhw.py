"""Figures cut from the HG-DAgger session recording.

Source: figures_repo/online_rl_hil/hil_ui/hgdagger_ui_real_robot.mp4, a 116 s
screen capture of the Online-RL page driving the AI Worker on 2026-08-28 --
the session that produced the 13-episode dataset.

State is read from the banner at x 766..814, y 281..297. In BGR a blue mean is
AUTO and an orange mean is HUMAN; getting that order backwards inverts the
whole timeline, so it is asserted against a known frame below.
"""
import cv2, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
rcParams.update({"font.family":"DejaVu Sans","font.size":8,"savefig.dpi":220})
HUM=(170/255,45/255,45/255); AUT=(30/255,95/255,175/255)

SRC="/home/robotis/figures_repo/online_rl_hil/hil_ui/hgdagger_ui_real_robot.mp4"
CAM=(0,233,60,1100)          #camera strip: y0,y1,x0,x1
cap=cv2.VideoCapture(SRC); FPS=cap.get(5); N=int(cap.get(7)); DUR=N/FPS

def frame(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES,min(int(t*FPS),N-1)); ok,f=cap.read()
    if not ok: raise RuntimeError(f"no frame at {t}s")
    return f

def state(f):
    b,g,r=[float(x) for x in f[281:297,766:814].reshape(-1,3).mean(0)]
    if r>190 and b<140: return "HUMAN"
    if b>180 and r<160: return "AUTO"
    return "?"

assert state(frame(60))=="HUMAN", "banner colours read backwards"
assert state(frame(45))=="AUTO",  "banner colours read backwards"

#full timeline at 0.5 s, for the operator spans
ts=np.arange(0,DUR,0.5)
st=[state(frame(t)) for t in ts]
spans=[]; s=None
for t,x in zip(ts,st):
    if x=="HUMAN" and s is None: s=t
    if x!="HUMAN" and s is not None: spans.append((s,t)); s=None
if s is not None: spans.append((s,ts[-1]))
print("operator spans:", [(round(a,1),round(b,1)) for a,b in spans])

# ---------------- filmstrip ----------------
def scrolled(f):
    """True when the page has scrolled and the camera strip is off-screen."""
    y0,y1,x0,x1=CAM
    return float(f[y0:y1,x0:x1].std()) < 28

SHOTS=[6,20,32,46,61,70,80,92,100,105]
ok=[]
for t in SHOTS:
    f=frame(t)
    if scrolled(f): print(f"  skip {t}s: page scrolled"); continue
    s_=state(f)
    if s_=="?":     print(f"  skip {t}s: banner mid-transition"); continue
    ok.append((t,f,s_))
print("using", [(t,s_) for t,_,s_ in ok])

rows=(len(ok)+1)//2
fig,ax=plt.subplots(rows,2,figsize=(7.4,rows*1.06),
                    gridspec_kw={"wspace":0.03,"hspace":0.40})
for a,(t,f,s_) in zip(ax.ravel(),ok):
    y0,y1,x0,x1=CAM
    a.imshow(cv2.cvtColor(f[y0:y1,x0:x1],cv2.COLOR_BGR2RGB))
    a.set_xticks([]); a.set_yticks([])
    col = HUM if s_=="HUMAN" else AUT
    for sp in a.spines.values(): sp.set_color(col); sp.set_linewidth(1.4)
    a.set_title(f"{t} s   {s_}",fontsize=7.5,color=col,pad=2.5)
for a in ax.ravel()[len(ok):]: a.axis("off")
fig.savefig("figs/hardware_rollout.png",bbox_inches="tight",facecolor="white")
plt.close(fig); print("wrote figs/hardware_rollout.png")

# ---------------- UI states ----------------
def save_ui(t,name):
    f=frame(t)
    cv2.imwrite(f"figs/{name}.png",f)
    print(f"wrote figs/{name}.png  (t={t}s, {state(f)})")

save_ui(46,"ui_state_auto")
save_ui(62,"ui_state_human")
cap.release()
