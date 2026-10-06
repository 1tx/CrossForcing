import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import torch, torch.nn.functional as F
import config
from data import build
from model import DeltaOperator

def metrics(pred, truth):
    p = pred.reshape(-1); t = truth.reshape(-1)
    m = torch.isfinite(p) & torch.isfinite(t)
    p, t = p[m], t[m]
    rmse = torch.sqrt(F.mse_loss(p, t)).item()
    mae = torch.mean(torch.abs(p - t)).item()
    bias = (p.mean() - t.mean()).item()
    vp = p - p.mean(); vt = t - t.mean()
    r = (vp*vt).sum()/(vp.norm()*vt.norm()+1e-12)
    return rmse, mae, bias, r.item()

mc = config.cfg.model
pack = build("era5")
target = pack["target"]; forcing = pack["forcing"]
static_t = torch.from_numpy(pack["static"]).float()
n, T, _ = target.shape
(sp_s, sp_e) = pack["split"]["spinup"]
(te_s, te_e) = pack["split"]["test"]

model = DeltaOperator(mc)
model.load_state_dict(torch.load(config.CACHE_DIR/"model_delta.pt", map_location="cpu", weights_only=True))
model.eval()

state = torch.from_numpy(target[:, sp_e-1].copy()).float()
print("init state shape", state.shape, flush=True)
fr = np.zeros((n, te_e-te_s, 2), dtype=np.float32)
for i, t in enumerate(range(te_s, te_e)):
    fb = torch.from_numpy(forcing[:, t].copy()).float()
    with torch.no_grad():
        state, _ = model(state, fb, static_t)
    fr[:, i] = state.numpy()
    if (i+1) % 60 == 0:
        print(f"day {i+1}/{te_e-te_s}  surf_mean={state[:,0].mean().item():.4f}", flush=True)

truth = target[:, te_s:te_e]
for j, name in enumerate(["surface", "rootzone"]):
    p = torch.from_numpy(fr[:, :, j]); t = torch.from_numpy(truth[:, :, j].copy())
    rmse, mae, bias, r = metrics(p, t)
    print(f"{name}: RMSE={rmse:.5f} MAE={mae:.5f} bias={bias:.5f} R={r:.4f}", flush=True)
print("FREERUN_OK")
