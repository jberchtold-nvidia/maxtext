import csv
import sys

if len(sys.argv) < 4:
    print("Usage: normalize.py input_raw_results.csv output_summary.{csv|txt} format")
    print("  format = 'csv' for comma-separated, 'txt' or 'tsv' for tab-separated")
    sys.exit(1)

input_csv = sys.argv[1]
output_file = sys.argv[2]
format_type = sys.argv[3].lower()

data = {}
key_order = []  # preserve order of keys

# Read input TSV
with open(input_csv) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        key = tuple(
            row.get(k) if row.get(k) not in [None, ""] else "NA"
            for k in ["dp", "tp", "tpsp", "fsdp"]
        )
        if not row.get("test"):
            continue
        if key not in data:
            data[key] = {}
            key_order.append(key)  # remember when first seen
        data[key][row["test"]] = row

header = ["test", "dp", "tp", "tpsp", "fsdp", "mean", "stddev", "normalized"]
rows = []

# iterate keys in first-seen order
for key in key_order:
    rowset = data[key]
    baseline = rowset.get("maxtext_fp8", {})
    base_mean = baseline.get("mean", "NA")
    try:
        base_mean_val = float(base_mean)
        has_baseline = True
    except Exception:
        has_baseline = False

    # iterate tests in first-seen order
    for testname in rowset:
        row = rowset[testname]
        mean = row["mean"]
        stddev = row["stddev"]
        if mean == "NA":
            normalized = "-"
        elif testname == "maxtext_fp8":
            normalized = "1.000" if has_baseline else "-"
        elif has_baseline and mean != "NA":
            try:
                normalized = f"{float(mean)/base_mean_val:.3f}"
            except Exception:
                normalized = "-"
        else:
            normalized = "-"
        rows.append(
            [
                testname,
                row["dp"],
                row["tp"],
                row["tpsp"],
                row["fsdp"],
                mean,
                stddev,
                normalized,
            ]
        )

if format_type in ("csv",):
    with open(output_file, "a", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(rows)
elif format_type in ("txt", "tsv"):
    with open(output_file, "a") as out:
        out.write("\t".join(header) + "\n")
        for r in rows:
            out.write("\t".join(r) + "\n")
else:
    print("Invalid format type! Use 'csv' or 'txt'/'tsv'.")
    sys.exit(2)

print(f"Done. Wrote summary to {output_file} as {format_type}.")
