# deps: pip install folium
import pandas as pd, numpy as np, ast, math, random
import folium


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

# ---------- hex math (flat-top hex tiling) ----------
# Choose a standard O-RU cell profile (ISD in meters). Hex edge size s = ISD / 1.5
def hex_size_from_profile(profile="UMi"):
    profiles = {
        "UMi": 200.0,   # Urban Micro typical inter-site distance (~200 m)
        "UMa": 500.0,   # Urban Macro (~500 m)
    }
    isd = profiles.get(profile, 200.0)
    s = isd / 1.5      # hex edge length (also circumradius center->vertex), meters
    return s, isd

def deg_m_factors(lat_ref):
    M_PER_DEG_LAT = 111_320.0
    M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(lat_ref))
    return M_PER_DEG_LAT, M_PER_DEG_LON

def to_local_m(lat, lon, lat_ref, lon_ref, M_PER_DEG_LAT, M_PER_DEG_LON):
    x = (lon - lon_ref) * M_PER_DEG_LON
    y = (lat - lat_ref) * M_PER_DEG_LAT
    return x, y

def to_latlon_m(x, y, lat_ref, lon_ref, M_PER_DEG_LAT, M_PER_DEG_LON):
    lat = lat_ref + y / M_PER_DEG_LAT
    lon = lon_ref + x / M_PER_DEG_LON
    return lat, lon

# Axial <-> world (flat-top) — from Red Blob Games conventions
# Forward (axial q,r) -> meters (x,y)
def axial_to_xy(q, r, s):
    x = s * (1.5 * q)
    y = s * (math.sqrt(3) * (r + 0.5 * q))
    return x, y

# Inverse: meters (x,y) -> fractional axial (qf, rf)
def xy_to_axial(x, y, s):
    qf = (2.0 / 3.0) * (x / s)
    rf = (y / (s * math.sqrt(3))) - 0.5 * qf
    return qf, rf

# Cube rounding to nearest hex index (q,r)
def axial_round(qf, rf):
    xf = qf
    zf = rf
    yf = -xf - zf
    xr = round(xf)
    yr = round(yf)
    zr = round(zf)
    dx = abs(xr - xf)
    dy = abs(yr - yf)
    dz = abs(zr - zf)
    if dx > dy and dx > dz:
        xr = -yr - zr
    elif dy > dz:
        yr = -xr - zr
    else:
        zr = -xr - yr
    return int(xr), int(zr)  # axial (q,r)

# Hex polygon vertices around a center (x0,y0) with edge size s
def hex_vertices_xy(x0, y0, s):
    verts = []
    for a in (0, 60, 120, 180, 240, 300):
        rad = math.radians(a)
        vx = x0 + s * math.cos(rad)
        vy = y0 + s * math.sin(rad)
        verts.append((vx, vy))
    return verts




# ---------- draw ONLY selected trips; tile hexes; keep only those that contain points ----------
def plot_selected_trips_with_nonoverlap_cells(df_trips, n=200, profile="UMi",
                                              m_orus=50, metric="points",
                                              center=None, score_kind="median",
                                              outfile="porto_hex_selected_topm_center.html"):
    """
    1) Select the n trips that are 'mostly near' the city center:
       - center=(lat,lon) if provided, else dataset centroid
       - score = median (or mean) distance-to-center per trip in meters
       - pick n trips with smallest score
    2) Build a non-overlapping flat-top hex grid and assign each selected point to a hex.
    3) Keep only top m_orus hex cells by 'points' or 'trips' coverage and draw them.

    metric: "points" or "trips"
    score_kind: "median" or "mean"
    """
    assert metric in ("points", "trips")
    assert score_kind in ("median", "mean")
    if len(df_trips) == 0:
        raise ValueError("No trips to draw.")
    if m_orus <= 0:
        raise ValueError("m_orus must be > 0")

    # -------- center of city (lat, lon) --------
    if center is None:
        all_lat, all_lon = [], []
        for poly in df_trips["POLYLINE"]:
            if poly:
                a = np.asarray(poly, float)  # [:,0]=lon, [:,1]=lat
                all_lat.append(a[:,1]); all_lon.append(a[:,0])
        if not all_lat:
            raise ValueError("No points in df_trips.")
        center_lat = float(np.mean(np.concatenate(all_lat)))
        center_lon = float(np.mean(np.concatenate(all_lon)))
    else:
        center_lat, center_lon = float(center[0]), float(center[1])

    M_LAT, M_LON = deg_m_factors(center_lat)

    # -------- score each trip by distance-to-center --------
    scores = []
    for idx, row in df_trips.iterrows():
        poly = row["POLYLINE"]
        if not poly:
            continue
        a = np.asarray(poly, float)  # [:,0]=lon, [:,1]=lat
        dx = (a[:,0] - center_lon) * M_LON
        dy = (a[:,1] - center_lat) * M_LAT
        d  = np.hypot(dx, dy)  # meters
        score = np.median(d) if score_kind == "median" else np.mean(d)
        scores.append((idx, score))

    # pick the n trips with smallest score
    scores.sort(key=lambda x: x[1])
    keep_rows = [i for i, _ in scores[:min(n, len(scores))]]
    sample = df_trips.loc[keep_rows].reset_index(drop=True)

    # -------- collect points from selected trips --------
    pts_lat = []
    pts_lon = []
    trip_idx = []
    for j, poly in enumerate(sample["POLYLINE"].tolist()):
        if not poly:
            continue
        a = np.asarray(poly, float)
        pts_lat.append(a[:,1])
        pts_lon.append(a[:,0])
        trip_idx.append(np.full(len(a), j, dtype=int))
    if not pts_lat:
        raise ValueError("Selected trips have no points.")
    lat_arr = np.concatenate(pts_lat)
    lon_arr = np.concatenate(pts_lon)
    trip_idx = np.concatenate(trip_idx)

    # reference for hex math = mean of selected trips
    lat_ref = float(lat_arr.mean())
    lon_ref = float(lon_arr.mean())
    M_LAT_sel, M_LON_sel = deg_m_factors(lat_ref)

    # local meters for hex assignment
    xs = (lon_arr - lon_ref) * M_LON_sel
    ys = (lat_arr - lat_ref) * M_LAT_sel

    # hex size from profile (no-overlap tiling)
    s, isd = hex_size_from_profile(profile)
    print(f"Center: ({center_lat:.5f}, {center_lon:.5f}) | profile={profile} (ISD≈{isd:.0f} m) | hex edge s={s:.1f} m")
    print(f"Selected {len(sample)} trips nearest the center (score={score_kind}).")

    # assign each point to a unique hex
    q_list, r_list = [], []
    for x, y in zip(xs, ys):
        qf, rf = xy_to_axial(x, y, s)
        q, r = axial_round(qf, rf)
        q_list.append(q)
        r_list.append(r)
    q_arr = np.array(q_list, dtype=int)
    r_arr = np.array(r_list, dtype=int)

    # coverage counts per cell
    cell_keys = np.stack([q_arr, r_arr], axis=1)
    unique_cells, inv = np.unique(cell_keys, axis=0, return_inverse=True)
    points_per_cell = np.bincount(inv)

    trips_sets = {}
    for ci, t in zip(inv, trip_idx):
        if ci not in trips_sets: trips_sets[ci] = set()
        trips_sets[ci].add(int(t))
    trips_per_cell = np.array([len(trips_sets.get(i, set())) for i in range(len(unique_cells))], dtype=int)

    # choose top-m cells
    scores_cells = points_per_cell if metric == "points" else trips_per_cell
    order = np.argsort(-scores_cells)
    top_idx = order[:min(m_orus, len(order))]
    top_cells = unique_cells[top_idx]
    top_scores = scores_cells[top_idx]

    frame = LocalMeterFrame.from_sample(sample)
    # build ORUs & hex polygons
    orus = []
    hex_polys_latlon = []
    for q, r in top_cells:
        cx, cy = axial_to_xy(q, r, s)
        lat_c, lon_c = to_latlon_m(cx, cy, lat_ref, lon_ref, M_LAT_sel, M_LON_sel)
        orus.append(ORU(np.array((lat_c, lon_c), dtype=float),frame))
        verts_xy = hex_vertices_xy(cx, cy, s)
        verts_latlon = [to_latlon_m(vx, vy, lat_ref, lon_ref, M_LAT_sel, M_LON_sel) for (vx, vy) in verts_xy]
        hex_polys_latlon.append(verts_latlon)

    # -------- draw map --------
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # draw selected trips + start/end
    for _, row in sample.iterrows():
        poly = row["POLYLINE"]  # [[lon,lat], ...]
        if not poly: continue
        latlon = [(pt[1], pt[0]) for pt in poly]
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        folium.PolyLine(latlon, weight=2, opacity=0.9, color=color).add_to(m)
        folium.CircleMarker(latlon[0], radius=3, color="green", fill=True,
                            tooltip=f"START | TAXI_ID={row['TAXI_ID']}").add_to(m)
        folium.CircleMarker(latlon[-1], radius=3, color="red", fill=True,
                            tooltip=f"END | TAXI_ID={row['TAXI_ID']}").add_to(m)

    # draw chosen hexes and O-RUs
    for verts, score in zip(hex_polys_latlon, top_scores):
        folium.Polygon(
            locations=[(la, lo) for (la, lo) in verts],
            color="#3388ff", weight=1, opacity=0.85, fill=True, fill_opacity=0.12,
            tooltip=f"Selected cell | score={int(score)} ({metric})"
        ).add_to(m)
    for o in orus:
        folium.CircleMarker(
            location=(float(o.position[0]), float(o.position[1])),
            radius=3, color="blue", fill=True, fill_opacity=1.0,
            tooltip="O-RU"
        ).add_to(m)

    m.save(outfile)
    print(f"Saved map: {outfile} | trips shown: {len(sample)} | O-RUs placed: {len(orus)} | metric={metric} | score={score_kind}")
    users=create_users_from_sample(sample, frame)
    return users, orus



class User:
    """
    A mobile user that replays a single trip trajectory.
    - trajectory_latlon: list/array of (lat, lon) points (in order).
    - next_move(): advance exactly one point; returns new (lat, lon) or None if finished.
    """
    def __init__(self, trajectory_latlon, frame: LocalMeterFrame, user_id=None):
        traj = np.asarray(trajectory_latlon, dtype=float)  # shape (T,2), (lat, lon)
        if traj.ndim != 2 or traj.shape[1] != 2 or len(traj) == 0:
            raise ValueError("trajectory_latlon must be a non-empty sequence of (lat, lon) points.")
        self.user_id = user_id
        self.trajectory = traj
        self.frame = frame
        self.idx = 0
        self.position = self.trajectory[0].copy()
        self.position_m = np.array(self.frame.to_m(*self.position), dtype=float)

    def next_move(self):
        """Advance one point along the trajectory. Return (lat, lon) or None when trip ends."""
        if self.idx + 1 >= len(self.trajectory):
            return None
        self.idx += 1
        self.position = self.trajectory[self.idx].copy()
        self.position_m = np.array(self.frame.to_m(*self.position), dtype=float)
        return tuple(self.position)

    def has_next(self):
        return (self.idx + 1) < len(self.trajectory)

    def reset(self):
        self.idx = 0
        self.position = self.trajectory[0].copy()
        self.position_m = np.array(self.frame.to_m(*self.position), dtype=float)


def create_users_from_sample(sample_df, frame: LocalMeterFrame, min_points=2):
    """
    Convert the selected trips (sample_df) into a list of Users.
    Each row's POLYLINE -> one User whose trajectory is that trip.

    - min_points: require at least this many points per trip.
    - use_taxi_id: store TAXI_ID (if present) as user_id; otherwise uses the row index.
    """
    users = []
    for idx, row in sample_df.iterrows():
        poly = row["POLYLINE"]
        if not poly or len(poly) < min_points:
            continue
        # POLYLINE is [[lon, lat], ...]; convert to (lat, lon)
        traj_latlon = [(p[1], p[0]) for p in poly]
        uid = int(idx)
        users.append(User(traj_latlon, user_id=uid,frame=frame))
    return users

# -------- run it --------
# n = number of trips to draw; profile="UMi" (≈200 m ISD) or "UMa" (≈500 m ISD)
# Place only 40 O-RUs that maximize coverage of the selected trips by number of points
users, o_rus = plot_selected_trips_with_nonoverlap_cells(
                    df_clean,
                    n=200,
                    profile="UMa",        # or "UMa"
                    m_orus=40,
                    metric="points",      # or "trips"
                    center=None,          # or (41.1579, -8.6291) for Porto city center
                    score_kind="median",  # "median" is robust; "mean" is okay too
                    outfile="porto_hex_topm_center.html"
                )

from core.environment import *
from core.utils import channel_gain
import cvxpy as cp
import time

num_orus = len(o_rus)
total_users =len(users)

B_budget_khz = 5000.0     # per-O-RU bandwidth budget [kHz]
T_max = 100.00               # per-user latency threshold [s]
n_bits = np.full(total_users, 2e9)  # 2000 kb = 2000000000 bits for cifar10

# Power budgets (set to what you want)
P_budget_per_oru = 1.5    # example total power budget per O-RU

# Noise spectral density (W/Hz). ~ -174 dBm/Hz ≈ 3.98e-21 W/Hz
# Use your system's N0; do NOT reuse a "total noise power" here.
N0_spec = 4.0e-21


near_rt_time_span = 1
global_round_threshold = 100
total_time = near_rt_time_span * global_round_threshold
energy_J = 0
rate_per_user_bps = np.zeros(total_users)
channel_gains = np.empty((global_round_threshold, num_orus, total_users))

for t in range(global_round_threshold):
    for u in range(total_users):
        users[u].next_move()

    for b in range(num_orus):
        for u in range(total_users):
            channel_gains[t, b, u] = channel_gain(o_rus[b], users[u])

# ---------- Variables ----------
max_iter = int(global_round_threshold / near_rt_time_span)
for gt in range(max_iter):
    print(f"Near-RT RIC is performing {gt}")
    # Precompute alpha = g / N0 (unit: 1/W·Hz)
    alpha = channel_gains[gt * near_rt_time_span] / N0_spec    # shape (B,U)
    # ---------- Constraints ----------
    cons = []

    B = cp.Variable((num_orus, total_users), nonneg=True)
    P = cp.Variable((num_orus, total_users), nonneg=True)
    Rnat = cp.Variable((num_orus, total_users), nonneg=True)
    t = cp.Variable(total_users, nonneg=True)
    cons += [t <= 1]
    # Per-O-RU bandwidth budgets
    cons += [cp.sum(B[b, :]) <= B_budget_khz          for b in range(num_orus)]
    cons += [cp.sum(P[b, :]) <= P_budget_per_oru      for b in range(num_orus)]

    # Shannon rate via exponential cone:
    B_Hz = 1e3 * B
    Z    = B_Hz + cp.multiply(alpha, P)          # same shape as B
    # Vectorized ExpCone: applies elementwise across the arrays
    cons += [cp.constraints.ExpCone(Rnat, B_Hz, Z)]

    # Per-user latency requirement: sum_b R[b,u] >= n_bits[u]/T_max  (convert bits/s -> NATs/s)
    required_rate_bps = n_bits / T_max
    required_rate_nat = required_rate_bps * np.log(2.0)     # NAT/s

    cons += [cp.sum(Rnat, axis=0) >= cp.multiply(required_rate_nat,t)]

    # ---------- Objective: maximize total sum-rate ----------
    lambda_energy = 1
    total_energy = cp.sum(P)
    objective = cp.Maximize(cp.sum(t)-lambda_energy*total_energy)

    # ---------- Solve ----------
    start_time = time.time()
    prob = cp.Problem(objective, cons)
    # Prefer MOSEK (fast & accurate on exp cones); fallback to SCS if needed
    prob.solve(solver=cp.MOSEK, verbose=False)

    print("time: %.2f s" % (time.time() - start_time))
    print("status:", prob.status)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("Infeasible or unbounded")
    else:
        B_val = B.value
        P_val = P.value
        Rnat_val = Rnat.value
        Rbps_val = Rnat_val / np.log(2.0)  # per-(b,u) rate in bits/s

        connections = [[] for b in range(num_orus)]
        connections_users = [[] for u in range(total_users)]
        for b in range(num_orus):
            for u in range(total_users):
                if B_val[b,u]>10:
                    connections[b].append((u,B_val[b,u]))
                    connections_users[u].append(b)

        for t in range(near_rt_time_span):
            print(f"=============================={gt*near_rt_time_span+t}====================================")
            counter= 0

            for b in range(num_orus):
                num_users_per_oru = len(connections[b])
                channel_gains_oru = np.empty((num_users_per_oru))
                c = 0
                for (u,r) in connections[b]:
                    channel_gains_oru[c] = channel_gains[gt * near_rt_time_span + t,b,u]
                    c+=1

                B = cp.Variable((num_users_per_oru), nonneg=True)
                P = cp.Variable((num_users_per_oru), nonneg=True)
                Rnat = cp.Variable((num_users_per_oru), nonneg=True)
                s = cp.Variable(num_users_per_oru, nonneg=True)

                cons = []
                # Per-O-RU bandwidth budgets
                cons += [(cp.sum(B)) <= B_budget_khz]
                cons += [(cp.sum(P)) <= P_budget_per_oru]

                # alpha = channel_gains / N0_spec  # shape (B,U)

                g = np.clip(channel_gains_oru, 0.0, None)  # no negative gains
                N0 = max(float(N0_spec), 1e-18)  # > 0
                alpha = g / N0_spec

                n_bits_oru = np.array([n_bits[u] for (u,r) in connections[b]])
                B_Hz = 1e3 * (B)
                Z = B_Hz + cp.multiply(alpha, P)  # same shape as B
                # Vectorized ExpCone: applies elementwise across the arrays
                eps = 1e-9
                Z_scale_guess = float(1e3 * B_budget_khz) + float(np.max(alpha) * P_budget_per_oru)
                S = max(1.0, Z_scale_guess)
                cons += [cp.constraints.ExpCone(Rnat/ S, B_Hz/ S, Z / S)]

                # Per-user latency requirement: sum_b R[b,u] >= n_bits[u]/T_max  (convert bits/s -> NATs/s)
                required_rate_bps = n_bits_oru / T_max
                required_rate_nat = required_rate_bps * np.log(2.0)  # NAT/s

                cons += [Rnat + s >= required_rate_nat+500]

                # ---------- Objective: maximize total sum-rate ----------
                T_slot = T_max
                lambda_R = 0.1
                lambda_viol = 1
                total_energy = cp.sum(P)
                viol_penalty = cp.sum(s)
                objective = cp.Minimize(total_energy + lambda_viol * viol_penalty)

                # ---------- Solve ----------
                start_time = time.time()
                prob = cp.Problem(objective, cons)

                prob.solve(solver=cp.MOSEK, verbose=False)

                print("time: %.2f s" % (time.time() - start_time))
                print("status:", prob.status)
                if prob.status not in ("optimal", "optimal_inaccurate"):
                    print("Infeasible or unbounded")
                else:
                    print("Optimal solution found")

                B_val = B.value
                P_val = P.value
                Rnat_val = Rnat.value
                Rbps_val = Rnat_val / np.log(2.0)
                energy_J += float(P_val.sum())
                u_prim = 0
                for connection in connections[b]:
                    u = connection[0]

                    rate_per_user_bps[u] += Rbps_val[u_prim]

                    latency_per_user_s = n_bits[u] / np.maximum(Rbps_val[u_prim], 1e-12)
                    meets = latency_per_user_s <= T_max + 1e-2
                    if meets:
                        counter +=1
                    # print(f"User {u:2d}: Rate={Rbps_val[u_prim]:12.2f} bps, bandwidth={B_val[u_prim]}  "
                    #       f"Lat={latency_per_user_s :.6f} s  Meets? {meets}")

                    n_bits[u] = max(n_bits[u] - Rbps_val[u_prim], 0)
                    u_prim +=1

            print(f"total successful users: {counter}")
            T_max = T_max - 1



for u in range(total_users):
    meets = n_bits[u] <= 10
    print(f"User {u:2d}: {n_bits[u]} Rate={rate_per_user_bps[u] / global_round_threshold:12.2f} bps  Meets? {meets}")

print(f"total successful users: {np.bincount(n_bits <= 10)[1]}")
print(f"Energy : {energy_J:.6f} J")


