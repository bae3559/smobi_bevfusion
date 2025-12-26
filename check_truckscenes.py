import os, math
from collections import defaultdict

from truckscenes.truckscenes import TruckScenes
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO

DATA_ROOT = "/home/bevfusion/data/man-truckscenes"
VERSION = 'v1.0-mini'

OUT_DIR = "./cat_samples_2"
N_PER_CAT = 3
CAM_KEY = "CAM_FRONT"

GRID_COLS=2



os.makedirs(OUT_DIR, exist_ok=True)

ts = TruckScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)

def make_collage(images, cols=2):
    if not images:
        return None
    
    w = min(im.width for im in images)
    h = min(im.height for im in images)
    images = [im.resize((w,h)) for im in images]
    rows = math.ceil(len(images)/cols)
    print(rows)
    canvas = Image.new("RGB", (w*cols,h*rows))
    for idx, im in enumerate(images) :
        r,  c = divmod(idx, cols)
        canvas.paste(im, (c*w, r*h))
    return canvas

#2) 카테고리별로 sample_token 을 "중복 없이" 모으기
cat_to_samples = defaultdict(list)
seen = defaultdict(set)
cat_to_ann_by_sample = defaultdict(lambda:defaultdict(list))

for ann in ts.sample_annotation:
    cname= ann["category_name"]
    stoken=ann["sample_token"]

    if stoken not in seen[cname]:
        seen[cname].add(stoken)
        cat_to_samples[cname].append(stoken)
    cat_to_ann_by_sample[cname][stoken].append(ann["token"])

#3) 카테고리마다 3개씩 렌더링 & 저장


saved = 0

for cname in sorted(cat_to_samples.keys()):

    # pick = cat_to_samples[cname][:N_PER_CAT]
    import random
    random.seed(0)
    samples= cat_to_samples[cname]
    pick=random.sample(samples, k=min(N_PER_CAT, len(samples)))

    for i, stoken in enumerate(pick, 1):
        sample = ts.get("sample", stoken)

        # CAM_KEY 가 없으면 첫 camera key로 fallback
        cam_keys = [k for k in sample["data"].keys() if str(k).upper().startswith("CAM")]
        if not cam_keys:
            continue

        cam = CAM_KEY if CAM_KEY in sample["data"] else cam_keys[0]
        sd_token = sample["data"][cam]

        # render_sample_data 는 내부적으로 matplotlib figure를 띄우는 경우가 많아서 
        # 저장하려면 현재 figure를 잡아서 저장하면 됨. 

        original_anns = list(sample["anns"])
        sample["anns"] = cat_to_ann_by_sample[cname][stoken]


        rendered_images=[]
        for cam in cam_keys:
            sd_token = sample["data"][cam]
            ts.render_sample_data(sd_token, with_anns=True)

            fig = plt.gcf()
            ax0 = fig.axes[0] if fig.axes else plt.gca()
            for ax in fig.axes:
                ax.set_title("")
                ax.set_axis_off()
                ax.set_position([0,0,1,1])

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            

            ax0.text(
                0.01, 0.99, cam,
                transform = ax0.transAxes,
                ha="left", va="top", 
                fontsize=14, color="white", 
                bbox=dict(facecolor="black", alpha=0.5, pad=2)
            )
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            img = Image.open(buf).convert("RGB")
            rendered_images.append(img)

        sample["anns"] = original_anns

        collage = make_collage(rendered_images, cols=GRID_COLS)
        if collage is not None:
            fname= f"{safe_name(cname)}_stoken-{stoken}_collage.png"
            collage.save(os.path.join(OUT_DIR, fname))
            saved +=1
        print(f"{cname} sample_token={stoken}")
print("done saving")