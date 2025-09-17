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


ckpt = torch.load("eapp_mlp.pt", map_location="cpu")

# rebuild model with the saved hyperparams
seq_len = 10
model = DeepMLPNextPos(seq_len=seq_len).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()




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

from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
# ---------- draw ONLY selected trips; tile hexes; keep only those that contain points ----------
def plot_selected_trips_with_nonoverlap_cells(df_trips, n=200, profile="UMi",
                                              m_orus=50, metric="points",
                                              window_m = 5000,frac_in_window=1.0,
                                              center=None, score_kind="median",
                                              k_groups = 5,
                                              outfile="porto_hex_selected_topm_center.html"):
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
    # -------- select trips by window around center --------
    keep_rows = []

    for idx, row in df_trips.iterrows():
        poly = row["POLYLINE"]
        if not poly:
            continue
        a = np.asarray(poly, float)  # [:,0]=lon, [:,1]=lat
        dx = (a[:, 0] - center_lon) * M_LON
        dy = (a[:, 1] - center_lat) * M_LAT
        d = np.hypot(dx, dy)  # meters to center

        # fraction of trip points inside the radius
        inside_frac = float(np.mean(d <= float(window_m)))
        if inside_frac >= float(frac_in_window):
            keep_rows.append(idx)

    # if too many qualify, keep the first n (or random sample if desired)
    if len(keep_rows) > n:
        rng = np.random.default_rng(42)
        keep_rows = rng.choice(keep_rows, size=n, replace=False).tolist()

    sample = df_trips.loc[keep_rows].reset_index(drop=True)
    print(f"Center window radius = {window_m} m, kept {len(sample)} trips (≥ {frac_in_window:.2f} inside).")

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
    # ----- build ORUs & hex polygons -----
    orus = []
    hex_polys_latlon = []
    oru_xy = []  # << collect (x,y) in meters for clustering

    for q, r in top_cells:
        cx, cy = axial_to_xy(q, r, s)
        oru_xy.append((cx, cy))  # store meter coords
        lat_c, lon_c = to_latlon_m(cx, cy, lat_ref, lon_ref, M_LAT_sel, M_LON_sel)
        orus.append(ORU(np.array((lat_c, lon_c), dtype=float), frame))

        verts_xy = hex_vertices_xy(cx, cy, s)
        verts_latlon = [to_latlon_m(vx, vy, lat_ref, lon_ref, M_LAT_sel, M_LON_sel) for (vx, vy) in verts_xy]
        hex_polys_latlon.append(verts_latlon)

    oru_polys = [Polygon([(lo, la) for (la, lo) in verts])  # shapely expects (x=lon, y=lat)
                 for verts in hex_polys_latlon]



    oru_xy = np.asarray(oru_xy, dtype=float)  # shape (N_oru, 2)
    oru_latlon = np.array([o.position for o in orus], dtype=float)  # (N_oru, 2) (lat, lon)

    # ----- choose number of groups -----
    # k_groups = 5 # or set explicitly, e.g., k_groups = 5

    # ----- cluster in meter-space -----
    kmeans = KMeans(n_clusters=k_groups, n_init='auto', random_state=0)
    o_ru_clusters = kmeans.fit_predict(oru_xy)  # shape (N_oru,)

    group_regions = {}
    for g in range(k_groups):
        polys_g = [oru_polys[i] for i, lab in enumerate(o_ru_clusters) if lab == g]
        if len(polys_g) == 0:
            group_regions[g] = None
        elif len(polys_g) == 1:
            group_regions[g] = polys_g[0]
        else:
            group_regions[g] = unary_union(polys_g)  # MultiPolygon or Polygon

    # simple color palette (extend if needed)
    palette = [
        "#292921", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
        "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3"
    ]
    get_color = lambda i: palette[i % len(palette)]

    # ----- draw map -----
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # trips
    for _, row in sample.iterrows():
        poly = row["POLYLINE"]
        if not poly: continue
        latlon = [(pt[1], pt[0]) for pt in poly]
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        folium.PolyLine(latlon, weight=2, opacity=0.9, color=color).add_to(fmap)
        folium.CircleMarker(latlon[0], radius=3, color="green", fill=True,
                            tooltip=f"START | TAXI_ID={row['TAXI_ID']}").add_to(fmap)
        folium.CircleMarker(latlon[-1], radius=3, color="red", fill=True,
                            tooltip=f"END | TAXI_ID={row['TAXI_ID']}").add_to(fmap)

    # hexes (optional: tint by cluster of their center O-RU)
    for verts, score, lab in zip(hex_polys_latlon, top_scores, o_ru_clusters):
        folium.Polygon(
            locations=[(la, lo) for (la, lo) in verts],
            color=get_color(lab), weight=1, opacity=0.85, fill=True, fill_opacity=0.10,
            tooltip=f"Selected cell | score={int(score)} ({metric}) | group={lab}"
        ).add_to(fmap)

    # O-RU markers colored by cluster
    for (lat, lon), lab in zip(oru_latlon, o_ru_clusters):
        folium.CircleMarker(
            location=(float(lat), float(lon)),
            radius=4, color=get_color(lab), fill=True, fill_opacity=1.0,
            tooltip=f"O-RU | group={lab}"
        ).add_to(fmap)


    users = create_users_from_sample(sample, frame,eApp=model)

    user_latlon_list = [ (u.position[0], u.position[1]) for u in users]

    # 3) Assign each user to a group
    user_group = np.full(len(users), -1, dtype=int)

    for i, (lat, lon) in enumerate(user_latlon_list):
        pt = Point(lon, lat)  # (x=lon, y=lat)
        assigned = None
        # first: membership by containment/touch
        for g, region in group_regions.items():
            if region is None:
                continue
            if region.contains(pt) or region.touches(pt):
                assigned = g
                break
        # fallback: nearest region (e.g., user outside all regions)
        if assigned is None:
            dists = [region.distance(pt) if region is not None else np.inf
                     for region in group_regions.values()]
            assigned = int(np.argmin(dists))
        user_group[i] = assigned

    # 5) Draw users on the Folium map, colored by their group
    palette = [
        "#292921", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
        "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3"
    ]
    get_color = lambda i: palette[i % len(palette)]

    for (lat, lon), g in zip(user_latlon_list, user_group):
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            color=get_color(g),
            fill=True, fill_opacity=0.9,
            tooltip=f"User | group={g}"
        ).add_to(fmap)  # or .add_to(m) if your map var is 'm'

    fmap.save(outfile)
    print(
        f"Saved map: {outfile} | trips shown: {len(sample)} | O-RUs placed: {len(orus)} | metric={metric} | score={score_kind}")
    return users, orus, group_regions, o_ru_clusters




def create_users_from_sample(sample_df, frame: LocalMeterFrame,eApp, min_points=2):

    users = []
    for idx, row in sample_df.iterrows():
        poly = row["POLYLINE"]
        if not poly or len(poly) < min_points:
            continue
        # POLYLINE is [[lon, lat], ...]; convert to (lat, lon)
        traj_latlon = [(p[1], p[0]) for p in poly]
        uid = int(idx)
        users.append(User(traj_latlon, user_id=uid,frame=frame, eApp = eApp,seq_len=seq_len))
    return users




from core.environment import *
from core.utils import channel_gain
import cvxpy as cp
import time

def mac_scheduling(connections, channel_gains, channel_gains_real, total_p_val, t, T_max , n_bits, coef, importance_weight, orus_by_idx, lambda_e=0):
    energy_J = 0
    counter = 0
    avg_computation_time = 0
    computation_counter = 0
    b_prim = 0
    for b in orus_by_idx:
        num_users_per_oru = len(connections[b_prim])
        if num_users_per_oru > 0:
            computation_counter +=1
            channel_gains_oru = np.empty((num_users_per_oru))
            channel_gains_oru_real = np.empty((num_users_per_oru))
            u_prim = 0

            for (u, r) in connections[b_prim]:
                channel_gains_oru[u_prim] = channel_gains[b, u]
                channel_gains_oru_real[u_prim] = channel_gains_real[b, u]
                u_prim += 1

            B = cp.Variable((num_users_per_oru), nonneg=True)
            P = cp.Variable((num_users_per_oru), nonneg=True)
            Rnat = cp.Variable((num_users_per_oru), nonneg=True)
            s = cp.Variable(num_users_per_oru, nonneg=True)

            cons = []
            cons += [s<=1]
            cons += [(cp.sum(B)) <= B_budget_khz]
            cons += [P <= total_p_val[b][np.asarray(connections[b_prim], dtype=int)[:,0]]]


            alpha = (channel_gains_oru / N0_spec) / 1e6

            n_bits_oru = np.array([n_bits[u] for (u, r) in connections[b_prim]])
            T_max_oru = np.asarray([T_max[u] for (u, r) in connections[b_prim]])
            B_Hz = B
            Z = (B_Hz + cp.multiply(alpha, P))  # same shape as B
            cons += [cp.constraints.ExpCone(Rnat, B_Hz, Z)]

            # Per-user latency requirement: sum_b R[b,u] >= n_bits[u]/T_max  (convert bits/s -> NATs/s)
            coef_per_oru = coef[np.asarray(connections[b_prim], dtype=int)[:,0]]
            importance_weight_oru = importance_weight[np.asarray(connections[b_prim], dtype=int)[:,0]]
            required_rate_bps = (coef_per_oru * n_bits_oru) / T_max_oru
            required_rate_nat = required_rate_bps * np.log(2.0) / 1e6  # NAT/s


            total_energy = cp.sum(P)
            cons += [Rnat >= cp.multiply(required_rate_nat,s)]
            objective = cp.Maximize(cp.sum(cp.multiply(importance_weight_oru, s))-total_energy*lambda_e)

            # ---------- Solve ----------
            start_time = time.time()
            prob = cp.Problem(objective, cons)
            prob.solve(solver=cp.MOSEK, verbose=False)
            avg_computation_time += time.time() - start_time
            print(time.time() - start_time)
            print("status:", prob.status)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                print("Infeasible or unbounded")
            else:
                print("Optimal solution found")

            P_val = P.value
            Rbps_val = (Rnat.value / np.log(2.0)) * 1e6
            energy_J += float(P_val.sum())*0.1

            B_Hz = np.asarray(B.value)  # Hz
            P_W = np.asarray(P.value)  # W

            g = channel_gains_oru_real  # no negative gains
            alpha = g / N0_spec
            alpha = alpha / 1e6
            R_cap = B_Hz * np.log1p(np.divide(alpha * P_W, B_Hz, out=np.zeros_like(B_Hz), where=B_Hz > 0))


            R_cap = (R_cap / np.log(2.0)) * 1e6

            mask = T_max_oru <= 1e-5
            R_cap = np.where(mask, 0.0, R_cap)
            u_prim = 0
            for connection in connections[b_prim]:
                u = connection[0]
                rate_per_user_bps[u] += R_cap[u_prim]
                latency_per_user_s = n_bits[u] / np.maximum(R_cap[u_prim], 1e-12)
                meets = latency_per_user_s <= T_max[u] + 1
                if meets:
                    counter += 1

                n_bits[u] = max(n_bits[u] - R_cap[u_prim] * 0.1, 0)
                u_prim += 1
        b_prim += 1
    if computation_counter>0:
        avg_computation_time/=computation_counter
    return energy_J, counter, avg_computation_time


def mobility_handler(channel_gains, num_orus, num_users, users_per_odu, orus_per_odu, lambda_e=0):

    alpha = channel_gains[np.ix_(orus_per_odu,users_per_odu)] / N0_spec  # shape (B,U)
    alpha = alpha / 1e6
    # ---------- Constraints ----------
    cons = []

    B = cp.Variable((num_orus, num_users), nonneg=True)
    P = cp.Variable((num_orus, num_users), nonneg=True)
    Rnat = cp.Variable((num_orus, num_users), nonneg=True)
    t = cp.Variable(num_users, nonneg=True)
    cons += [t <= 1]
    # Per-O-RU bandwidth budgets
    cons += [cp.sum(B[b, :]) <= B_budget_khz for b in range(num_orus)]

    cons += [cp.sum(P[:, u]) <= P_budget_per_user for u in range(num_users)]
    # Shannon rate via exponential cone:
    B_Hz = B
    Z = B_Hz + cp.multiply(alpha, P)  # same shape as B
    # Vectorized ExpCone: applies elementwise across the arrays
    cons += [cp.constraints.ExpCone(Rnat, B_Hz, Z)]

    # Per-user latency requirement: sum_b R[b,u] >= n_bits[u]/T_max  (convert bits/s -> NATs/s)
    n_bits_odu = n_bits[users_per_odu]
    T_max_odu =  T_max[users_per_odu]
    coef_odu = coef[users_per_odu]

    required_rate_bps = (coef_odu * n_bits_odu) / T_max_odu
    required_rate_nat = required_rate_bps * np.log(2.0) / 1e6  # NAT/s

    cons += [(cp.sum(Rnat, axis=0)) >= cp.multiply(required_rate_nat, t)]

    # ---------- Objective: maximize total sum-rate ----------
    lambda_energy = 0.0
    total_energy = cp.sum(P)
    importance_weight_odu = importance_weight[users_per_odu]
    objective = cp.Maximize(cp.sum(cp.multiply(t, importance_weight_odu))-lambda_e*total_energy)

    # ---------- Solve ----------
    prob = cp.Problem(objective, cons)
    # Prefer MOSEK (fast & accurate on exp cones); fallback to SCS if needed
    prob.solve(solver=cp.MOSEK, verbose=False)

    print("status:", prob.status)

    B_val = B.value * 1e6
    P_val = P.value
    Rbps_val = (Rnat.value / np.log(2.0))* 1e6  # per-(b,u) rate in bits/s

    return prob.status, B_val, P_val, Rbps_val



def cross_odu_assignment(group_regions, users):
    user_latlon_list = [(u.position[0], u.position[1]) for u in users]

    # 3) Assign each user to a group
    user_group = np.full(len(users), -1, dtype=int)

    for i, (lat, lon) in enumerate(user_latlon_list):
        pt = Point(lon, lat)  # (x=lon, y=lat)
        assigned = None
        # first: membership by containment/touch
        for g, region in group_regions.items():
            if region is None:
                continue
            if region.contains(pt) or region.touches(pt):
                assigned = g
                break
        # fallback: nearest region (e.g., user outside all regions)
        if assigned is None:
            dists = [region.distance(pt) if region is not None else np.inf
                     for region in group_regions.values()]
            assigned = int(np.argmin(dists))
        user_group[i] = assigned
    return user_group



# -------- run it --------
# n = number of trips to draw; profile="UMi" (≈200 m ISD) or "UMa" (≈500 m ISD)
# Place only 40 O-RUs that maximize coverage of the selected trips by number of points
num_odus = 4
save = True
if save:
    users, o_rus, group_regions, o_ru_clusters = plot_selected_trips_with_nonoverlap_cells(
                        df_clean,
                        n=150,
                        profile="UMa",        # or "UMa"
                        m_orus=20,
                        window_m = 1000,
                        metric="points",      # or "trips"
                        center=None,          # or (41.1579, -8.6291) for Porto city center
                        score_kind="median",  # "median" is robust; "mean" is okay too
                        k_groups= num_odus,
                        outfile="porto_hex.html"
                    )

    save_pickle({"users": users, "orus": o_rus, "group_regions":group_regions, "o_ru_clusters":o_ru_clusters}, "users_orus.pkl")
else:
    data = load_pickle("users_orus.pkl")
    users = data["users"]
    o_rus = data["orus"]
    group_regions = data["group_regions"]
    o_ru_clusters = data["o_ru_clusters"]



num_orus = len(o_rus)
total_users =len(users)
num_services = 3
user_permutation = np.random.permutation(total_users)
users1, users2, users3 = np.split(user_permutation, num_services)

idx_by_service = [users1, users2, users3]
service_of_user = np.full(total_users, -1, dtype=int)
for s, idx in enumerate(idx_by_service):
    service_of_user[idx]= s


# Power budgets (set to what you want)
P_budget_per_user = 0.3 # W
# Noise spectral density (W/Hz). ~ -174 dBm/Hz ≈ 3.98e-21 W/Hz
# Use your system's N0; do NOT reuse a "total noise power" here.
N0_spec = 4.0e-21
T_max_service = np.asarray([20, 20, 20], dtype=float)  # per-service latency threshold [s]
T_max = T_max_service[service_of_user]
time_frame = 0.1
near_rt_time_span = 40
global_round_threshold = int(np.max(T_max * (1/time_frame)))
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

rate_per_user_bps = np.zeros(total_users)

num_global_rounds = 100



# ---------- Variables ----------
max_iter = int(global_round_threshold / near_rt_time_span)
channel_gains = np.zeros((num_global_rounds, global_round_threshold, num_orus, total_users))
channel_gains_real = np.zeros((num_global_rounds, global_round_threshold, num_orus, total_users))

num_oru_per_odu = np.empty((num_global_rounds, max_iter), dtype=object)
num_user_per_odu = np.empty((num_global_rounds, max_iter), dtype=object)
idx_oru_by_group = np.empty((num_global_rounds, max_iter), dtype=object)
idx_user_by_group = np.empty((num_global_rounds, max_iter), dtype=object)


save = True
if save:
    for gr in range(num_global_rounds):
        for gt in range(max_iter):
            user_group = cross_odu_assignment(group_regions, users)
            num_oru_per_odu[gr, gt] = np.bincount(o_ru_clusters, minlength=num_odus)
            num_user_per_odu[gr, gt] = np.bincount(user_group, minlength=num_odus)
            idx_oru_by_group[gr, gt] = [np.where(o_ru_clusters == g)[0] for g in range(num_odus)]
            idx_user_by_group[gr, gt] = [np.where(user_group == g)[0] for g in range(num_odus)]

            for t in range(near_rt_time_span):
                time_stamp = gt * near_rt_time_span + t
                for u in range(total_users):
                    users[u].next_move()

                for b in range(num_orus):
                    for u in range(total_users):
                        channel_gains[gr,time_stamp][b, u], channel_gains_real[gr,time_stamp][b, u] = channel_gain(o_rus[b], users[u])

    save_pickle({"channel_gains": channel_gains, "channel_gains_real": channel_gains_real,
                 'num_oru_per_odu':num_oru_per_odu, 'num_user_per_odu':num_user_per_odu,
                 'idx_oru_by_group':idx_oru_by_group, 'idx_user_by_group':idx_user_by_group,
                 }, "channel_gains.pkl")
else:
    data = load_pickle("channel_gains.pkl")
    channel_gains = data["channel_gains"]
    channel_gains_real = data["channel_gains_real"]
    num_oru_per_odu= data["num_oru_per_odu"]
    num_user_per_odu = data["num_user_per_odu"]
    idx_oru_by_group = data["idx_oru_by_group"]
    idx_user_by_group = data["idx_user_by_group"]


for lambda_e in [0, 1,2,4,6,8,10]:
    success_status = np.empty((num_global_rounds, num_services), dtype=object)
    energy_statistics = np.zeros((num_global_rounds))
    near_rt_latency_statistics = np.zeros((num_global_rounds, max_iter, num_odus))
    mac_latency_statistics = np.zeros((num_global_rounds, global_round_threshold))

    near_rt_latency_statistics_per_gr = np.zeros((num_global_rounds))
    mac_latency_statistics_per_gr = np.zeros((num_global_rounds))

    for gr in range(num_global_rounds):
        print(f"++++++++++++++++++++++++++++++ Global round {gr} ++++++++++++++++++++++++++++++++++")
        energy_J = 0
        B_budget_khz = 5.0  # per-O-RU bandwidth budget [kHz]
        T_max_service = np.asarray([20, 20, 20], dtype=float)  # per-service latency threshold [s]
        T_max = T_max_service[service_of_user]
        coef = np.full(total_users, 1, dtype=int)

        n_bits_service = np.asarray([1e8, 1e8, 1e8])  # 2e7 b = 2.5 MB bits for cifar10
        n_bits = n_bits_service[service_of_user]

        importance_weight_service = np.asarray([1, 1, 1])
        importance_weight = importance_weight_service[service_of_user]

        for gt in range(max_iter):
            for d in range(num_odus):
                print(f"=============================={gt * near_rt_time_span}====================================")
                print(f"Near-RT RIC of O-DU {d} is performing {gt}")
                # Precompute alpha = g / N0 (unit: 1/W·Hz)
                if num_user_per_odu[gr, gt][d]>0 and num_oru_per_odu[gr, gt][d]>0:
                    start_time = time.time()
                    status, B_val, P_val, Rbps_val = mobility_handler(channel_gains = channel_gains[gr, gt * near_rt_time_span], num_orus = num_oru_per_odu[gr, gt][d],
                                                                      num_users = num_user_per_odu[gr, gt][d], users_per_odu = idx_user_by_group[gr, gt][d],
                                                                      orus_per_odu = idx_oru_by_group[gr,gt][d], lambda_e=lambda_e)
                    P_val_total = np.zeros((num_orus, total_users))
                    p_prim = 0
                    for b in idx_oru_by_group[gr, gt][d]:
                        P_val_total[b][idx_user_by_group[gr, gt][d]] = P_val[p_prim]
                        p_prim += 1
                    near_rt_latency_statistics[gr, gt, d] = time.time() - start_time
                    near_rt_latency_statistics_per_gr[gr] += time.time() - start_time
                    print("Near_rt computation time: %.2f s" % (near_rt_latency_statistics[gr, gt, d]))
                    T_max[idx_user_by_group[gr, gt][d]] -= near_rt_latency_statistics[gr, gt, d]

                    if status not in ("optimal", "optimal_inaccurate"):
                        print("Infeasible or unbounded")
                    else:
                        connections = [[] for b in range(num_oru_per_odu[gr, gt][d])]
                        connections_users = [[] for u in range(num_user_per_odu[gr, gt][d])]
                        b_prim = 0
                        for b in idx_oru_by_group[gr, gt][d]:
                            u_prim = 0
                            for u in idx_user_by_group[gr, gt][d]:
                                if B_val[b_prim,u_prim] > 100 and P_val[b_prim,u_prim] > 0.01:
                                    connections[b_prim].append((u,B_val[b_prim,u_prim]))
                                    connections_users[u_prim].append(b)
                                u_prim+=1
                            b_prim+=1

                        for t in range(near_rt_time_span):
                            print(f"====={gt*near_rt_time_span+t}====")
                            e, counter, computation_time = mac_scheduling(connections, channel_gains = channel_gains[gr, gt * near_rt_time_span + t], channel_gains_real= channel_gains_real[gr, gt * near_rt_time_span + t], total_p_val= P_val_total,
                                                                          t= gt * near_rt_time_span + t, T_max = T_max, n_bits = n_bits, coef = coef, importance_weight= importance_weight,
                                                                          orus_by_idx = idx_oru_by_group[gr, gt][d], lambda_e=lambda_e)
                            mac_latency_statistics[gr, gt * near_rt_time_span + t] = computation_time
                            print("MAC scheduler computation time: %.2f s" % (mac_latency_statistics[gr, gt * near_rt_time_span + t]))

                            mac_latency_statistics[gr] +=computation_time
                            energy_J +=e
                            print(f"total successful users: {counter}")
                            T_max[idx_user_by_group[gr, gt][d]] = np.maximum(T_max[idx_user_by_group[gr, gt][d]] - max(time_frame, computation_time), 1e-5)
                            mask = T_max  <= 1e-5
                            coef = np.where(mask, 0.0, coef)


        for u in range(total_users):
            meets = n_bits[u] <= 10
            print(f"User {u:2d}: {n_bits[u]} Rate={rate_per_user_bps[u] / global_round_threshold:12.2f} bps  Meets? {meets}")

        if len(np.bincount(n_bits <= 10)) == 2:
            bb = np.bincount(n_bits <= 10)[1]
        else:
            bb = 0
        print(f"total successful users: {bb}")
        print(f"Energy : {energy_J:.6f} J")

        energy_statistics[gr] = energy_J

        for s, idxs in enumerate(idx_by_service):
            successful_user_service = (n_bits[idxs] <= 10)
            success_status[gr, s] = successful_user_service
            print(f"success status of service {s} = {success_status[gr, s]}")
            if len(np.bincount(success_status[gr, s]))==2:
                bb = np.bincount(success_status[gr, s])[1]
            else:
                bb = 0
            print(f"total successful users of service {s}: {b}")

        save_pickle({"success_status": success_status}, f"results/EVFL/success_status_{near_rt_time_span}_{lambda_e}.pkl")
        save_pickle({"energy_statistics": energy_statistics}, f"results/EVFL/energy_statistics_{near_rt_time_span}_{lambda_e}.pkl")
        save_pickle({"mac_latency_statistics": mac_latency_statistics_per_gr}, f"results/EVFL/mac_latency_statistics_{near_rt_time_span}_{lambda_e}.pkl")
        save_pickle({"near_rt_latency_statistics": near_rt_latency_statistics_per_gr},f"results/EVFL/near_rt_latency_statistics_{near_rt_time_span}_{lambda_e}.pkl")


