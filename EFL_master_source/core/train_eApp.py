# deps: pip install folium
import pandas as pd, numpy as np, ast, math, random
import folium
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import random_split, DataLoader, Subset
from torch.utils.data import Dataset
from tqdm import tqdm

from core.neural_networks import DeepMLPNextPos
from environment import *
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "mps")






# ----------- Minimal PyTorch dataset for next-15s displacement -----------

def sample_trips_in_window(df_trips, n_trips, window_m, frame, frac_in_window=0.6, center=None, seed=42):
    """
    Sample up to n_trips from df_trips, keeping only trips where at least
    frac_in_window of points lie within window_m meters of the city center.

    Uses LocalMeterFrame for lon/lat -> meters conversion:
      x = (lon - lon_ref) * M_LON,  y = (lat - lat_ref) * M_LAT
    """
    # ---- center (lat, lon) ----
    if center is None:
        all_lat, all_lon = [], []
        for poly in df_trips["POLYLINE"]:
            if poly:
                a = np.asarray(poly, float)  # [:,0]=lon, [:,1]=lat
                all_lat.append(a[:,1]); all_lon.append(a[:,0])
        if not all_lat:
            raise ValueError("No points in df_trips to infer center.")
        center_lat = float(np.mean(np.concatenate(all_lat)))
        center_lon = float(np.mean(np.concatenate(all_lon)))
    else:
        center_lat, center_lon = float(center[0]), float(center[1])

    # ---- filter trips by window ----
    keep_rows = []
    for idx, row in df_trips.iterrows():
        poly = row["POLYLINE"]
        if not poly:
            continue
        a = np.asarray(poly, float)                 # [:,0]=lon, [:,1]=lat
        x = (a[:,0] - frame.lon_ref) * frame.M_LON  # meters east (lon)
        y = (a[:,1] - frame.lat_ref) * frame.M_LAT  # meters north (lat)
        d = np.hypot(x, y)                          # distance to center (0,0) in meters

        inside_frac = float(np.mean(d <= float(window_m)))
        if inside_frac >= float(frac_in_window):
            keep_rows.append(idx)

    if len(keep_rows) == 0:
        raise ValueError("No trips fall inside the requested window. Increase window_m or lower frac_in_window.")

    # ---- sample up to n_trips from qualifying set ----
    rng = np.random.default_rng(seed)
    if len(keep_rows) > int(n_trips):
        keep_rows = rng.choice(keep_rows, size=int(n_trips), replace=False).tolist()

    return df_trips.loc[keep_rows].reset_index(drop=True)


class PortoNextStepDataset(Dataset):

    def __init__(self, df_trips, frame: LocalMeterFrame, seq_len: int = 30,
                 sub_dt_sec: float = 1.0, base_dt_sec: float = 15.0):
        assert seq_len >= 2, "seq_len must be >= 2"
        assert sub_dt_sec > 0.0 and base_dt_sec > 0.0, "time steps must be > 0"

        # how many equal sub-steps per original 15s segment (>=1)
        k = max(1, int(round(base_dt_sec / float(sub_dt_sec))))
        self.eff_dt_sec = base_dt_sec / k   # actual sub-step seconds after rounding
        self.seq_len = int(seq_len)

        X_list, y_list = [], []

        for poly in df_trips["POLYLINE"]:
            if not poly or len(poly) < 2:
                continue

            arr = np.asarray(poly, dtype=float)   # [:,0]=lon, [:,1]=lat]
            # lon/lat -> local meters (x=east from lon, y=north from lat)
            x = (arr[:,0] - frame.lon_ref) * frame.M_LON
            y = (arr[:,1] - frame.lat_ref) * frame.M_LAT
            xy = np.column_stack([x, y]).astype(np.float32)  # (T,2)

            # densify to k sub-steps between each pair
            xy = self._densify_k(xy, k)  # (T',2) with T' >= T

            if xy.shape[0] < self.seq_len + 1:
                continue

            # sliding windows on densified sequence
            T = xy.shape[0]
            for t in range(self.seq_len, T):
                past = xy[t-self.seq_len:t]     # (seq_len, 2)
                dxy  = xy[t] - xy[t-1]          # (2,) displacement over ~eff_dt_sec
                X_list.append(past)
                y_list.append(dxy)

        if not X_list:
            raise RuntimeError("No training windows were created. "
                               "Try reducing seq_len or ensure trips have enough points.")

        self.X = torch.from_numpy(np.stack(X_list, axis=0))  # (N, seq_len, 2), float32
        self.y = torch.from_numpy(np.stack(y_list, axis=0))  # (N, 2), float32

    @staticmethod
    def _densify_k(xy: np.ndarray, k: int) -> np.ndarray:
        """
        Insert (k-1) evenly spaced samples between each consecutive pair.
        k=1 returns the input unchanged.
        """
        if k <= 1 or xy.shape[0] < 2:
            return xy.astype(np.float32, copy=False)

        out = []
        for i in range(xy.shape[0] - 1):
            p0, p1 = xy[i], xy[i+1]
            if i == 0:
                out.append(p0)
            step = (p1 - p0) / float(k)
            # add k sub-steps to reach p1 exactly
            for s in range(1, k+1):
                out.append(p0 + s * step)
        return np.asarray(out, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def train_next_position_mlp(df_trips, frame, seq_len=10, epochs=100, batch_size=64, lr=1e-3, subset_k=None):
    """
    Train a tiny MLP on the PortoNextStepDataset to predict the next **position**.
    - df_trips: cleaned DataFrame with POLYLINE
    - subset_k: if set (e.g., 100_000), train on a random subset of K windows
    """
    # 1) Frame + dataset (dataset returns past window + next-step delta)
    full_ds = PortoNextStepDataset(df_trips, frame=frame, seq_len=seq_len)

    # Optional: train on a subset of windows
    ds = full_ds
    if subset_k is not None:
        N = len(full_ds)
        k = min(int(subset_k), N)
        idx = torch.randperm(N)[:k]
        ds = Subset(full_ds, idx)

    # 2) Split (10% val) and loaders
    N = len(ds)
    val_n = max(1, int(0.1 * N))
    tr_n = N - val_n
    tr_ds, va_ds = random_split(ds, [tr_n, val_n], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    # 3) Model / opt / loss

    model = DeepMLPNextPos(seq_len=seq_len).to(device)
    # model = ResMLPNextPos(seq_len=seq_len, width=512, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.SmoothL1Loss(beta=30.0)  # meters
    error1 = np.zeros((epochs))
    error2 = np.zeros((epochs))

    model.eval()
    init_tr_loss, seen = 0.0, 0
    with torch.no_grad():
        for xb, y_delta in tr_loader:
            xb = xb.to(device)
            y_delta = y_delta.to(device)
            last_pos = xb[:, -1, :]
            true_next = last_pos + y_delta
            pred_next = last_pos + model(xb)  # forward pass only
            loss = loss_fn(pred_next, true_next)
            bs = xb.size(0)
            init_tr_loss += float(loss.item()) * bs
            seen += bs
    init_tr_loss /= max(1, seen)
    print(f"Initial train loss (before training): {init_tr_loss:.4f}")

    # 4) Train
    for ep in range(1, epochs+1):
        model.train()
        tr_loss, seen = 0.0, 0
        for xb, y_delta in tqdm(tr_loader, desc="train", unit="batch"):            # xb: (B,T,2), y_delta: (B,2)
            xb = xb.to(device)
            y_delta = y_delta.to(device)

            last_pos = xb[:, -1, :]              # (B,2)
            true_next = last_pos + y_delta       # (B,2)

            pred_delta = model(xb)               # (B,2)
            pred_next = last_pos + pred_delta    # (B,2)

            loss = loss_fn(pred_next, true_next)
            opt.zero_grad(); loss.backward(); opt.step()

            bs = xb.size(0); tr_loss += float(loss.item()) * bs; seen += bs
        tr_loss /= max(1, seen)

        model.eval()
        va_loss, va_mae, seen = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, y_delta in tqdm(va_loader, desc="val", unit="batch"):
                xb = xb.to(device); y_delta = y_delta.to(device)
                last_pos = xb[:, -1, :]
                true_next = last_pos + y_delta
                pred_next = last_pos + model(xb)
                loss = loss_fn(pred_next, true_next)
                mae = torch.linalg.norm(pred_next - true_next, dim=1)
                bs = xb.size(0)
                va_loss += float(loss.item()) * bs
                va_mae  += float(mae.sum().item())
                seen += bs
        va_loss /= max(1, seen)
        va_mae  /= max(1, seen)
        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | val MAE {va_mae:.1f} m")
        error1[ep] = va_loss
        error2[ep] = tr_loss
        ckpt = {
            "model_state": model.state_dict(),
            "seq_len": 10,  # whatever you trained with
            "frame": {"lat_ref": frame.lat_ref, "lon_ref": frame.lon_ref},
        }
        torch.save(ckpt, "eapp_mlp.pt")
        save_pickle({"error1": error1},
                    f"results/EVFL/error1.pkl")
        save_pickle({"error1": error2},
                    f"results/EVFL/error2.pkl")
        print("saved -> eapp_mlp.pt")
    return model, ds


# ---------- helpers: parsing & trip quality ----------
def parse_poly(s):
    try:
        p = ast.literal_eval(s)
        return p if isinstance(p, list) else []
    except Exception:
        return []

def good_polyline(poly, thresh_m=500, min_points=2):
    """True if trip has >=min_points and no 15s step > thresh_m."""
    if not poly or len(poly) < min_points:
        return False
    a = np.asarray(poly, float)  # [:,0]=lon, [:,1]=lat
    if a.ndim != 2 or a.shape[1] != 2 or np.isnan(a).any():
        return False
    lat0 = float(a[:,1].mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat0))
    dlon = np.diff(a[:,0]); dlat = np.diff(a[:,1])
    step_m = np.hypot(dlon * m_per_deg_lon, dlat * m_per_deg_lat)
    return np.all(step_m <= thresh_m)


# ---------- load & keep only correct trips ----------
df = pd.read_csv(
    "train_sub.csv",
    usecols=["TAXI_ID", "POLYLINE", "TIMESTAMP", "MISSING_DATA"]
)
df["POLYLINE"] = df["POLYLINE"].map(parse_poly)
mask = (~df["MISSING_DATA"]) & df["POLYLINE"].map(lambda p: good_polyline(p, thresh_m=500))
df_clean = df.loc[mask, ["TAXI_ID", "TIMESTAMP", "POLYLINE"]].reset_index(drop=True)
print(f"Kept {len(df_clean)} / {len(df)} trips")

frame = LocalMeterFrame.from_sample(df_clean)
df_sub = sample_trips_in_window(df_clean, n_trips = 2000, window_m=1000, frame=frame)

model, ds_used = train_next_position_mlp(df_sub, frame = frame, seq_len=10, epochs=100, batch_size=64, lr=1e-3)


