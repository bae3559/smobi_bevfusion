# extract_camera_backbone_from_full_ckpt.py

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help ="pth파일 경로")
    parser.add_argument("--outdir", required=True, help="camera-only backbone 저장할 경로")
    args = parser.parse_args()


    # full_ckpt = "runs/waymo-camera-only/latest.pth"
    # out_ckpt  = "runs/waymo-camera-only/camera_backbone_only.pth"
    full_ckpt = args.checkpoint
    out_ckpt= args.outdir + "camera_backbone_only.pth"

    import torch, re

    ckpt = torch.load(full_ckpt, map_location='cpu')
    sd   = ckpt.get('state_dict', ckpt)

    # 1) prefix가 붙어있는 키만 필터
    prefix = "encoders.camera.backbone."
    keep = {k: v for k, v in sd.items() if k.startswith(prefix)}

    # 2) 백본 모듈 내부 키만 남기도록 prefix 제거
    remapped = { re.sub(rf"^{re.escape(prefix)}", "", k) : v for k, v in keep.items() }

    torch.save({'state_dict': remapped}, out_ckpt)
    print("saved:", out_ckpt, "kept keys:", len(remapped))

if __name__=="__main__":
    main()