import os

import numpy as np
import glob
import time

import tifffile

import tomopy

from bm3d_streak_removal import full_streak_pipeline, extreme_streak_attenuation, multiscale_streak_removal


PROJECTIONS_FILE_RANGE = 50
print(f"Running test_bm3d.py")
print(f"- working with {PROJECTIONS_FILE_RANGE} files")


ct_dir = "/HFIR/CG1D/IPTS-31148/raw/ct_scans/December3_2022"
assert os.path.exists(ct_dir)

ob_dir = "/HFIR/CG1D/IPTS-31148/raw/ob/December3_2022"
assert os.path.exists(ob_dir)

list_ct = glob.glob(os.path.join(ct_dir, "*.tiff"))
assert len(list_ct) > 0

list_ob = glob.glob(os.path.join(ob_dir, "*.tiff"))
assert len(list_ob) > 0


list_ct = list_ct[:PROJECTIONS_FILE_RANGE]

print(f"Loading projections ...", end="")
# loading projections
ct_data = []
for _file in list_ct:
    ct_data.append(tifffile.imread(_file))
ct_data = np.array(ct_data)
print(f" done!")

print(f"Loading ob ...", end="")
ob_data = []
for _file in list_ob:
    ob_data.append(tifffile.imread(_file))
ob_data = np.array(ob_data)
print(f" done!")

# normalization
print(f"normalization ...", end="")
ob_data_median = np.median(ob_data, axis=0)
ct_data_norm = []
for _data in ct_data:
    ct_data_norm.append(np.divide(_data, ob_data_median))
ct_data_norm = np.array(ct_data_norm)
print(f" done!")

# minus conversion
print(f"minus conversion ...", end="")
proj_m = tomopy.minus_log(ct_data_norm)
print(f" done!")

# algo 1
print(f"running bm3d extreme_streak_attenuation ...)
t_start = time.process_time()
bm3d_norm = extreme_streak_attenuation(proj_m)
t_end = time.process_time()
print(f"running bm3d extreme_streak_attenuation done in in {t_end - t_start}s!")

# algo 2
print(f"running bm3d multiscale_streak_removal ...")
t_start = time.process_time()
bm3d_denoised = multiscale_streak_removal(bm3d_norm)
t_end = time.process_time()
print(f"running bm3d multiscale_streak_removal done in {t_end - t_start}s!")

print(f"Done running test_bm3d!")