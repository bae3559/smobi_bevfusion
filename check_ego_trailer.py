import os, math
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from truckscenes.truckscenes import TruckScenes
TARGET = "vehicle.ego_trailer"
OUT_DIR = "./ego_trailer_all"
DATA_ROOT = "/home/bevfusion/data/man-truckscenes"
VERSION = 'v1.0-mini'
GRID_COLS = 2
os.makedirs(OUT_DIR, exist_ok=True)

ts = TruckScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)

def safe(s):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))

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

# ego_trailer가 있는 모든 sample_token
ego_anns = [a for a in ts.sample_annotation if a["category_name"] == TARGET]
ego_samples = sorted({a["sample_token"] for a in ego_anns})
print("samples:", len(ego_samples))

saved = 0

for stoken in ego_samples:
    sample = ts.get("sample", stoken)
    cam_keys = sorted([k for k in sample["data"].keys() if str(k).upper().startswith("CAM")])
    if not cam_keys:
        continue

    # 🔥 이 sample에서 ego_trailer ann token만 남기기
    ego_ann_tokens = [a["token"] for a in ego_anns if a["sample_token"] == stoken]
    if not ego_ann_tokens:
        continue

    original_anns = list(sample["anns"])
    sample["anns"] = ego_ann_tokens

    rendered_images = []

        
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
    # 원복
    sample["anns"] = original_anns

    collage = make_collage(rendered_images, cols=min(GRID_COLS, len(rendered_images)))
    if collage is not None:
        out_path = os.path.join(OUT_DIR, f"{safe(TARGET)}__{safe(stoken)}__collage.png")
        collage.save(out_path)
        saved += 1

print("saved collages:", saved, "->", OUT_DIR)
