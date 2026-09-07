import os
import pandas as pd
from collections import defaultdict
from multiscale_chirality_pipeline import match_file_to_code

labels_csv = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\cropped_em920_ex520-815\sample_labels.csv"
tp_dirs = [
    r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\cropped_em920_ex520-815\out_0h",
    r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\cropped_em920_ex520-815\out_6h",
    r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\cropped_em920_ex520-815\out_24h",
]

labels = pd.read_csv(labels_csv)
codes = labels["code"].astype(str).tolist()

for tp_dir in tp_dirs:
    mapping = {}
    for c in codes:
        fp = match_file_to_code(tp_dir, c)
        mapping[c] = os.path.basename(fp)

    inv = defaultdict(list)
    for c, f in mapping.items():
        inv[f].append(c)

    dups = {f: cs for f, cs in inv.items() if len(cs) > 1}
    print("\n", tp_dir)
    print("Duplicate filename count:", len(dups))
    for f, cs in list(dups.items())[:15]:
        print(" ", f, " <- ", cs)
