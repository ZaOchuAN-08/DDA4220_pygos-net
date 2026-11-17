#!/usr/bin/env python
"""End-to-end real-data smoke test using a deliberately small baseline CNN.
This is not full PygosNet: it validates real manifests, training, checkpoint
save/reload, test inference, CSV metrics and visual outputs.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from pygosnet.data.dataset import PenguinDataset

class SmallBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(16, 2, 1)
    def forward(self, image):
        size = (max(1, image.shape[-2] // 8), max(1, image.shape[-1] // 8))
        logits = self.head(self.features(image))
        logits = torch.nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return {"density": torch.nn.functional.softplus(logits[:, :1]),
                "segmentation_logits": logits[:, 1:2], "segmentation": torch.sigmoid(logits[:, 1:2])}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_dataset(root, manifest, split, args):
    path = root / manifest
    if not path.exists(): raise FileNotFoundError(f"Manifest not found: {path}")
    ds = PenguinDataset(path, root, split, args.crop_size, args.sigma, args.stride, args.support_threshold)
    if not len(ds): raise RuntimeError(f"Manifest is empty: {path}")
    limit = getattr(args, f"max_{split}_samples", 0)
    return Subset(ds, list(range(min(limit, len(ds))))) if limit and limit > 0 else ds

def run_epoch(model, loader, optimizer, device, train):
    model.train(train); sums = {"loss": 0., "density_loss": 0., "seg_loss": 0.}; n = 0
    for batch in loader:
        x = batch["image"].to(device); yd = batch["density"].to(device); ys = batch["seg_mask"].to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            out = model(x)
            if out["density"].shape != yd.shape:
                raise RuntimeError(f"Output/target shape mismatch: {tuple(out['density'].shape)} vs {tuple(yd.shape)}")
            dl = torch.nn.functional.smooth_l1_loss(out["density"], yd)
            sl = torch.nn.functional.binary_cross_entropy_with_logits(out["segmentation_logits"], ys)
            loss = dl + 0.1 * sl
            if train: loss.backward(); optimizer.step()
        b = x.shape[0]; n += b
        sums["loss"] += loss.detach().item()*b; sums["density_loss"] += dl.detach().item()*b; sums["seg_loss"] += sl.detach().item()*b
    return {k: v/max(n, 1) for k,v in sums.items()}

def save_preview(image_tensor, density, segmentation, path):
    from PIL import Image, ImageDraw
    a = (image_tensor.detach().cpu().clamp(0,1).numpy().transpose(1,2,0)*255).astype(np.uint8)
    image = Image.fromarray(a, "RGB"); w,h=image.size
    def panel(t):
        a=t.detach().cpu().numpy().squeeze().astype(np.float32); a=np.nan_to_num(a)
        if a.max()>a.min(): a=(a-a.min())/(a.max()-a.min())
        else: a=np.zeros_like(a)
        return Image.fromarray((a*255).astype(np.uint8)).convert("RGB").resize((w,h), Image.Resampling.NEAREST)
    canvas=Image.new("RGB",(3*w,h+26),"white")
    canvas.paste(image,(0,26)); canvas.paste(panel(density),(w,26)); canvas.paste(panel(segmentation),(2*w,26))
    d=ImageDraw.Draw(canvas); d.text((5,6),"Input",fill="black"); d.text((w+5,6),"Predicted density",fill="black"); d.text((2*w+5,6),"Segmentation",fill="black")
    canvas.save(path)

@torch.no_grad()
def test_export(model, loader, device, outdir, n_visuals):
    model.eval(); rows=[]; abs_err=[]; sq_err=[]; ious=[]
    visdir=outdir/"visualizations"; visdir.mkdir(parents=True,exist_ok=True)
    for batch in loader:
        x=batch["image"].to(device); out=model(x); pd=out["density"]; ps=out["segmentation"]
        masked=pd*(ps>=0.5); pc=masked.sum((1,2,3)); truth=batch["count"].to(device).reshape(-1); gtmask=batch["seg_mask"].to(device)>=0.5
        for i in range(x.shape[0]):
            p=float(pc[i].cpu()); t=float(truth[i].cpu()); e=p-t; abs_err.append(abs(e)); sq_err.append(e*e)
            pm=ps[i]>=0.5; inter=(pm&gtmask[i]).sum().item(); union=(pm|gtmask[i]).sum().item(); iou=1.0 if union==0 else inter/union; ious.append(iou)
            image_id=str(batch["image_id"][i]); rows.append({"image_id":image_id,"true_count":t,"predicted_count":p,"count_error":e,"absolute_count_error":abs(e),"density_sum_unmasked":float(pd[i].sum().cpu()),"segmentation_positive_fraction":float(pm.float().mean().cpu()),"segmentation_iou":iou})
            if len(rows)<=n_visuals:
                safe="".join(c if c.isalnum() or c in "-_." else "_" for c in image_id)
                save_preview(batch["image"][i],pd[i],ps[i],visdir/f"{safe}.png")
    if rows:
        with (outdir/"test_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    metrics={"test_images":len(rows),"count_MAE":float(np.mean(abs_err)) if abs_err else None,"count_RMSE":float(np.sqrt(np.mean(sq_err))) if sq_err else None,"mean_segmentation_IoU":float(np.mean(ious)) if ious else None,"note":"Exploratory results from a tiny baseline, not final PygosNet performance."}
    (outdir/"test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics

def main():
    p=argparse.ArgumentParser(description="Run the real-data pipeline with a small baseline CNN.")
    p.add_argument("--project-root",type=Path,default=Path.cwd()); p.add_argument("--train-manifest",default="data/splits/train.jsonl"); p.add_argument("--val-manifest",default="data/splits/val.jsonl"); p.add_argument("--test-manifest",default="data/splits/test.jsonl")
    p.add_argument("--output",default="outputs/real_data_baseline"); p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--crop-size",type=int,default=256)
    p.add_argument("--sigma",type=float,default=4.0); p.add_argument("--stride",type=int,default=8); p.add_argument("--support-threshold",type=float,default=1e-3); p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--max-train-samples",type=int,default=0); p.add_argument("--max-val-samples",type=int,default=0); p.add_argument("--max-test-samples",type=int,default=0); p.add_argument("--visualizations",type=int,default=12); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1: p.error("--epochs and --batch-size must be positive")
    root=a.project_root.resolve(); outdir=(root/a.output).resolve(); outdir.mkdir(parents=True,exist_ok=True); seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Project root: {root}\nDevice: {device}\nOutput: {outdir}")
    train=make_dataset(root,a.train_manifest,"train",a); val=make_dataset(root,a.val_manifest,"val",a); test=make_dataset(root,a.test_manifest,"test",a)
    print(f"Samples: train={len(train)}, val={len(val)}, test={len(test)}")
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers); val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers); test_loader=DataLoader(test,batch_size=1,shuffle=False,num_workers=a.num_workers)
    model=SmallBaseline().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5); history=[]; best=float("inf"); bestpath=outdir/"best_baseline.pt"; lastpath=outdir/"last_baseline.pt"
    for epoch in range(1,a.epochs+1):
        tr=run_epoch(model,train_loader,opt,device,True); va=run_epoch(model,val_loader,opt,device,False); rec={"epoch":epoch}|{f"train_{k}":v for k,v in tr.items()}|{f"val_{k}":v for k,v in va.items()}; history.append(rec)
        print(f"Epoch {epoch}/{a.epochs} | train loss={tr['loss']:.5f} | val loss={va['loss']:.5f}")
        ck={"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"config":{**vars(a),"project_root":str(root),"output":str(outdir)},"baseline_model":"SmallBaseline"}; torch.save(ck,lastpath)
        if va["loss"]<best: best=va["loss"]; torch.save(ck,bestpath)
    with (outdir/"history.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=list(history[0])); wr.writeheader(); wr.writerows(history)
    ck=torch.load(bestpath,map_location=device,weights_only=False); model2=SmallBaseline().to(device); model2.load_state_dict(ck["model"],strict=True); print(f"Checkpoint reload passed: {bestpath}")
    metrics=test_export(model2,test_loader,device,outdir,a.visualizations); print(json.dumps(metrics,indent=2,ensure_ascii=False))
    print("\nPIPELINE PASSED. Outputs: history.csv, best_baseline.pt, last_baseline.pt, test_predictions.csv, test_metrics.json, visualizations/")
    print("Reminder: this tiny baseline is for pipeline validation, not final model performance.")
if __name__=="__main__": main()
