#!/usr/bin/env python3
"""Write one W3 diversity CSV row. All values as positional args."""
import csv, sys

_, out, ds, sel, k, dst, pid, rpid, fscen, b0, b1, b2, ovh, marg, mis, notes = sys.argv
margin = float(marg)
ovh_f = float(ovh)
b0_f = float(b0)
b2_f = float(b2)
overhead_frac = f"{ovh_f/margin:.4f}" if margin > 0 else "NA"
allowed = "YES" if (margin > 0 and b2_f < b0_f and ovh_f < 0.9 * margin) else "NO"

with open(out, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([ds, int(sel), int(k), dst, int(pid), int(rpid), fscen,
                b0_f, float(b1), b2_f, ovh_f, margin, overhead_frac, allowed,
                int(mis), "CERTIFIED_BOUNDED", notes])
