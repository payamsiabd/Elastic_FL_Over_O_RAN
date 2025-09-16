import numpy as np
from matplotlib import pyplot as plt

from matplotlib.patches import RegularPolygon
random_gn = np.random.default_rng(45)
import math

import numpy as np, torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# class User:
#     def __init__(self, network_dimensions, pos=np.array((0, 0), dtype=float) , mobility_significant = 15):
#         self.position = pos
#         self.lv = LevyFlightMobilityModel(beta=5,sigma=1)
#         self.mobility_significant =  mobility_significant
#         self.network_dimensions = network_dimensions
#     def next_move(self):
#         next_move = self.position + self.lv.next_move()* self.mobility_significant
#         self.position = np.clip(next_move,self.network_dimensions[0], self.network_dimensions[1])
#
#
# class ORU:
#     def __init__(self, pos=np.array((0, 0), dtype=float)):
#         self.position = pos

class LocalMeterFrame:
    """
    City-scale local projection using one fixed origin (lat_ref, lon_ref).
    x = east (from longitude), y = north (from latitude), in meters.
    """
    def __init__(self, lat_ref: float, lon_ref: float):
        self.lat_ref = float(lat_ref)
        self.lon_ref = float(lon_ref)
        self.M_LAT = 111_320.0
        self.M_LON = self.M_LAT * math.cos(math.radians(self.lat_ref))

    @classmethod
    def from_sample(cls, sample_df):
        # Build frame centered at mean of all points in selected trips
        lats, lons = [], []
        for poly in sample_df["POLYLINE"]:
            if poly:
                arr = np.asarray(poly, float)  # [:,0]=lon, [:,1]=lat
                lats.append(arr[:,1]); lons.append(arr[:,0])
        lat_ref = float(np.mean(np.concatenate(lats)))
        lon_ref = float(np.mean(np.concatenate(lons)))


        return cls(lat_ref, lon_ref)

    def to_m(self, lat, lon):
        x = (float(lon) - self.lon_ref) * self.M_LON  # lon -> x (east)
        y = (float(lat) - self.lat_ref) * self.M_LAT  # lat -> y (north)
        return (x, y)

    def to_latlon(self, x_m, y_m):
        lat = self.lat_ref + float(y_m) / self.M_LAT
        lon = self.lon_ref + float(x_m) / self.M_LON
        return lat, lon


# ---------------- ORU class ----------------
class ORU:
    def __init__(self, pos=np.array((0, 0), dtype=float), frame: LocalMeterFrame=None):  # pos = [lat, lon]
        self.position = pos
        self.frame = frame
        self.position_m = None  # (x_m, y_m)
        if self.frame is not None:
            self.position_m = np.array(self.frame.to_m(*self.position), dtype=float)

    def set_frame(self, frame: LocalMeterFrame):
        """Attach/replace the local frame and (re)compute meters."""
        self.frame = frame
        self.position_m = np.array(self.frame.to_m(*self.position), dtype=float)

    def set_position(self, lat, lon):
        """Update lat/lon and keep meters in sync if a frame exists."""
        self.position = np.array((lat, lon), dtype=float)
        if self.frame is not None:
            self.position_m = np.array(self.frame.to_m(lat, lon), dtype=float)


class User:
    def __init__(self, trajectory_latlon, frame: LocalMeterFrame, eApp, seq_len: int, user_id=None):
        traj = np.asarray(trajectory_latlon, dtype=float)  # (T,2) as (lat, lon)
        if traj.ndim != 2 or traj.shape[1] != 2 or len(traj) == 0:
            raise ValueError("trajectory_latlon must be a non-empty sequence of (lat, lon) points.")
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2.")

        self.user_id = user_id
        self.frame = frame
        self.model = eApp.eval()  # inference mode
        self.seq_len = int(seq_len)
        self.segment_period_s = 15.0  # raw sampling period (seconds)


        # ----- REAL trajectory (lat/lon & meters) -----
        # Default playback = the original coarse paths (no interpolation yet)
        self.playback_real_xy_m = np.array([self.frame.to_m(lat, lon) for (lat, lon) in traj], dtype=np.float32)
        self.playback_real_latlon = traj.copy()

        # One shared pointer; we assume same dt when you interpolate both
        self.idx = 0
        self.set_interpolated_playbacks(dt_s=1) # MAC scheduler works every 100 second
        self.position_real = self.playback_real_latlon[0].copy()
        self.position_m_real = self.playback_real_xy_m[0].copy()

        # ----- PREDICTED trajectory (meters & lat/lon) -----
        T = self.playback_real_xy_m.shape[0]
        pred_xy = self.playback_real_xy_m.copy()  # init with real so first seq_len are valid
        if T >= self.seq_len + 1:
            with torch.no_grad():
                for t in range(self.seq_len, T):
                    window = self.playback_real_xy_m[t - self.seq_len:t]  # (seq_len,2)
                    xb = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)  # (1,seq,2)
                    xb = xb.to(device)
                    pred_delta = self.model(xb).squeeze(0).cpu().numpy()  # (2,)
                    pred_next = window[-1] + pred_delta
                    pred_xy[t] = pred_next  # store prediction at step t

        self.playback_pred_xy_m = pred_xy.astype(np.float32)  # (T,2)
        # back to lat/lon for convenience
        self.playback_pred_latlon = np.array([frame.to_latlon(px, py) for px, py in self.playback_pred_xy_m], dtype=float)

        # Current positions pulled from playback arrays
        self.position = self.playback_pred_latlon[0].copy()  # predicted current (lat, lon)
        self.position_m = self.playback_pred_xy_m[0].copy()  # predicted current (x_m, y_m)

        # print()

    # --- ADD this private helper inside the class ---
    def _interp_xy(self, P_xy_m: np.ndarray, dt_s: float, include_last: bool = True):
        """
        Linear interpolation in meters between consecutive 15s samples.
        Returns (xy_m_hi, latlon_hi) without storing time.
        """
        if dt_s <= 0:
            raise ValueError("dt_s must be > 0")
        T = P_xy_m.shape[0]
        if T < 2:
            # trivial: single point
            x_m, y_m = P_xy_m[0, 0], P_xy_m[0, 1]
            lat, lon = self.frame.to_latlon(x_m, y_m)
            return P_xy_m.copy(), np.array([[lat, lon]], float)

        total_dur = (T - 1) * self.segment_period_s
        tq = np.arange(0.0, total_dur + (1e-9 if include_last else 0.0), dt_s, dtype=float)
        if include_last and abs(tq[-1] - total_dur) > 1e-6:
            tq = np.append(tq, total_dur)

        seg = np.floor(tq / self.segment_period_s).astype(int)
        seg = np.clip(seg, 0, T - 2)
        tau = tq - seg * self.segment_period_s
        alpha = (tau / self.segment_period_s).reshape(-1, 1)  # (N,1)

        P0 = P_xy_m[seg]  # (N,2)
        P1 = P_xy_m[seg + 1]  # (N,2)
        xy_m_hi = (1.0 - alpha) * P0 + alpha * P1

        latlon_hi = np.empty_like(xy_m_hi, dtype=float)
        for i, (x_m, y_m) in enumerate(xy_m_hi):
            lat, lon = self.frame.to_latlon(x_m, y_m)
            latlon_hi[i, 0] = lat
            latlon_hi[i, 1] = lon
        return xy_m_hi.astype(np.float32), latlon_hi

    def set_interpolated_playbacks(self, dt_s: float, include_last: bool = True):
        real_xy, real_ll = self._interp_xy(self.playback_real_xy_m, dt_s, include_last)

        # Use the common length (they should match; if not, take the min to stay safe)
        L = len(real_xy)
        self.playback_real_xy_m = real_xy[:L]
        self.playback_real_latlon = real_ll[:L]

    # ---------- REAL replay API (unchanged) ----------
    def next_move(self):
        """Advance one step along BOTH interpolated (pred & real) playback paths."""
        L = len(self.playback_pred_latlon)
        if self.idx + 1 >= L:
            return None
        self.idx += 1

        # predicted
        self.position = self.playback_pred_latlon[self.idx].copy()
        self.position_m = self.playback_pred_xy_m[self.idx].copy()

        # real (parallel)
        self.position_real = self.playback_real_latlon[self.idx].copy()
        self.position_m_real = self.playback_real_xy_m[self.idx].copy()

        return tuple(self.position)  # returning predicted lat/lon as before

    def reset(self):
        self.idx = 0
        self.position = self.playback_pred_latlon[0].copy()
        self.position_m = self.playback_pred_xy_m[0].copy()
        self.position_real = self.playback_real_latlon[0].copy()
        self.position_m_real = self.playback_real_xy_m[0].copy()

    def has_next(self):
        return (self.idx + 1) < len(self.real_latlon)


    # ---------- Convenience: quick error stats ----------
    def mae_meters(self, start=None, end=None):

        if start is None: start = self.seq_len
        if end   is None: end   = len(self.real_xy_m)
        if end <= start:  return 0.0
        diff = self.pred_xy_m[start:end] - self.real_xy_m[start:end]
        return float(np.mean(np.linalg.norm(diff, axis=1)))


def hex_spiral_axial(n):
    if n <= 0: return []
    coords = [(0, 0)]
    if n == 1: return coords
    dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    k = 1
    while len(coords) < n:
        q, r = -k, +k
        for side in range(6):
            dq, dr = dirs[side]
            for _ in range(k):
                if len(coords) >= n: break
                coords.append((q, r))
                q += dq; r += dr
        k += 1
    return coords[:n]

def axial_to_xy_flat_top(q, r, size):
    x = size * (1.5 * q)
    y = size * (np.sqrt(3) * (r + 0.5 * q))
    return np.array([x, y], dtype=float)

def hex_network(num_orus, isd=300.0, rotate_deg=0.0):
    size = isd / np.sqrt(3.0)  # neighbor spacing = size*sqrt(3) = isd
    axial = hex_spiral_axial(num_orus)
    pts = np.array([axial_to_xy_flat_top(q, r, size) for (q, r) in axial], dtype=float)
    if rotate_deg:
        th = np.deg2rad(rotate_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts = pts @ R.T
    return pts, size

def hex_vertices(center, size):
    cx, cy = center
    angles = np.deg2rad(30 + 60*np.arange(6))  # flat-top
    return np.stack([cx + size*np.cos(angles), cy + size*np.sin(angles)], axis=1)

# =============== Sampling ===============

def sample_uniform_in_hex(center, size, n, rng=None, max_trials_per_point=80):
    if rng is None:
        rng = np.random.default_rng()
    verts = hex_vertices(center, size)
    edges = np.roll(verts, -1, axis=0) - verts
    normals = np.stack([edges[:,1], -edges[:,0]], axis=1)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    xmin, ymin = verts.min(axis=0); xmax, ymax = verts.max(axis=0)
    pts = []
    for _ in range(n):
        for _ in range(max_trials_per_point):
            p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
            rel = p - verts
            if np.all((rel * normals).sum(axis=1) <= 1e-12):
                pts.append(p); break
        else:
            pts.append(center)
    return np.array(pts)

def convex_hull(points):
    P = np.unique(points, axis=0)
    if len(P) <= 1: return P
    P = P[np.lexsort((P[:,1], P[:,0]))]
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1], dtype=float)

def sample_uniform_in_convex_polygon(poly, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    v0 = poly[0]
    tris = [(v0, poly[i], poly[i+1]) for i in range(1, len(poly)-1)]
    def tri_area(a,b,c): return 0.5 * abs(np.cross(b-a, c-a))
    areas = np.array([tri_area(a,b,c) for (a,b,c) in tris])
    probs = areas / areas.sum()
    idx = rng.choice(len(tris), size=n, p=probs)
    pts = []
    for k in idx:
        a,b,c = tris[k]
        r1, r2 = rng.random(), rng.random()
        u = 1 - np.sqrt(r1)
        v = np.sqrt(r1) * (1 - r2)
        w = np.sqrt(r1) * r2
        pts.append(u*a + v*b + w*c)
    return np.array(pts)

# =============== Pretty drawing utils ===============

def _apply_theme(ax, theme):
    if theme == "dark":
        ax.figure.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        fg = "#e6edf3"
        grid = "#30363d"
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        return dict(fg=fg, hex_edge="#414a58", hex_face="#1f6feb", user="#f78166", oru="#58a6ff")
    else:
        ax.figure.set_facecolor("white")
        ax.set_facecolor("white")
        fg = "#111111"
        grid = "#e6e6e6"
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        return dict(fg=fg, hex_edge="#9aa0a6", hex_face="#cfe8ff", user="#d43f3a", oru="#1b76d1")

def _add_glow_points(ax, x, y, s=140, color="#1b76d1", alpha=0.25, layers=3):
    # soft glow under points
    for k in range(layers, 0, -1):
        ax.scatter(x, y, s=s*(k*1.8), c=color, alpha=alpha*(0.5/k), linewidths=0, zorder=1)


def draw_hex_network_with_users(num_orus=19, users_total=95, isd=300.0, mode="cells",
                                rotate_deg=0.0, seed=1, theme="light",
                                show_labels=False, face_alpha=0.18):
    """
    mode: "cells" (uniform over union of hex cells) or "hull" (uniform over convex hull).
    theme: "light" or "dark".
    """
    rng = np.random.default_rng(seed)
    oru_pos, size = hex_network(num_orus, isd, rotate_deg)

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=110)
    colors = _apply_theme(ax, theme)

    # Hex cells with soft fill & crisp edges
    for (x, y) in oru_pos:
        ax.add_patch(
            RegularPolygon(
                (x, y), 6, radius=size,
                orientation=np.deg2rad(30),
                edgecolor=colors["hex_edge"], facecolor=colors["hex_face"],
                linewidth=1.2, alpha=face_alpha, zorder=0
            )
        )

    # Users
    if mode == "cells":
        counts = rng.multinomial(users_total, [1/num_orus]*num_orus)
        user_pts = []
        for b, cnt in enumerate(counts):
            if cnt > 0:
                user_pts.append(sample_uniform_in_hex(oru_pos[b], size, cnt, rng=rng))
        user_pts = np.vstack(user_pts) if user_pts else np.zeros((0,2))
        subtitle = "users uniform over union of hex cells"
    elif mode == "hull":
        all_verts = np.vstack([hex_vertices(c, size) for c in oru_pos])
        hull = convex_hull(all_verts)
        user_pts = sample_uniform_in_convex_polygon(hull, users_total, rng=rng)
        ax.fill(hull[:,0], hull[:,1], facecolor=colors["hex_face"], alpha=0.08,
                edgecolor=colors["hex_edge"], lw=1.0, zorder=0)
        subtitle = "users uniform over convex hull"
    else:
        raise ValueError("mode must be 'cells' or 'hull'")

    # ORU glow, then markers (gives a nice halo)
    _add_glow_points(ax, oru_pos[:,0], oru_pos[:,1], s=80, color=colors["oru"], alpha=0.35, layers=3)
    ax.scatter(oru_pos[:,0], oru_pos[:,1], s=60, c=colors["oru"], edgecolor="white",
               linewidths=1.0, marker="o", label="O-RUs", zorder=6)

    # Users (small x markers)
    ax.scatter(user_pts[:,0], user_pts[:,1], s=18, c=colors["user"], marker="x",
               linewidths=1.0, label="Users", zorder=4)

    # Optional labels
    if show_labels:
        for i, (x, y) in enumerate(oru_pos):
            ax.text(x, y, str(i), ha='center', va='center',
                    color="white" if theme=="dark" else "#0b0b0b",
                    fontsize=8, weight='bold', zorder=7)

    # Framing
    ax.set_aspect('equal', adjustable='box')
    margin = 2.1 * size
    ax.set_xlim(oru_pos[:,0].min() - margin, oru_pos[:,0].max() + margin)
    ax.set_ylim(oru_pos[:,1].min() - margin, oru_pos[:,1].max() + margin)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Hex network: {num_orus} O-RUs  |  {users_total} Users  |  ISD={isd} m\n{subtitle}",
                 fontsize=12, pad=10)

    # Prettier legend
    leg = ax.legend(frameon=False, loc='upper right')
    for lh in leg.legendHandles:
        try: lh.set_alpha(1.0)
        except: pass


    plt.tight_layout()
    plt.show()
    return oru_pos, user_pts, (np.array([oru_pos[:,0].min() - isd/2, oru_pos[:,1].min() - isd/2]), np.array([ oru_pos[:,0].max() + isd/2, oru_pos[:,1].max() + isd/2]))