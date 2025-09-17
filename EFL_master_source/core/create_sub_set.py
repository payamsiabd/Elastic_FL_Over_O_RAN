# make_porto_subset.py
# Usage:
#   python make_porto_subset.py --in_csv train.csv --out_csv porto_subset.csv \
#       --sample_size 100000 --seq_len 20 --seed 42
#
# Keeps only rows with enough points (>= seq_len+1) and MISSING_DATA == False
# Uses reservoir sampling to pick a uniform random subset across the full file.

import argparse, ast, csv, random
import pandas as pd

def parse_polyline(s):
    try:
        pts = ast.literal_eval(s)
        return pts if isinstance(pts, list) else []
    except Exception:
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default='train.csv')
    ap.add_argument("--out_csv", default='train_sub.csv')
    ap.add_argument("--sample_size", type=int, default=100_000)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    random.seed(args.seed)
    min_len = args.seq_len + 1

    # Columns we’ll preserve (minimal set; add more if you need them)
    keep_cols = ["TRIP_ID","CALL_TYPE","ORIGIN_CALL","ORIGIN_STAND","TAXI_ID",
                 "TIMESTAMP","DAY_TYPE","MISSING_DATA","POLYLINE"]

    # Create writer with header
    out_f = open(args.out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=keep_cols)
    writer.writeheader()

    # Reservoir
    reservoir = []
    seen = 0

    # Stream the big CSV
    for chunk in pd.read_csv(args.in_csv, usecols=keep_cols, chunksize=args.chunksize):
        for _, row in chunk.iterrows():
            # Filter eligibility
            if bool(row.get("MISSING_DATA", False)):
                continue
            pts = parse_polyline(row["POLYLINE"])
            if len(pts) < min_len:
                continue

            # Eligible row: reservoir-sample it
            seen += 1
            if len(reservoir) < args.sample_size:
                reservoir.append(row.to_dict())
            else:
                j = random.randrange(seen)
                if j < args.sample_size:
                    reservoir[j] = row.to_dict()

    # Write reservoir to out_csv
    for r in reservoir:
        writer.writerow(r)
    out_f.close()

    print(f"Wrote {len(reservoir)} rows to {args.out_csv} (eligible seen={seen})")

if __name__ == "__main__":
    main()
