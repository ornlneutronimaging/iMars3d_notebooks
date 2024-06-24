import numpy as np


def convert_time_s_in_time_hr_mn_s(time_s):
    time_s_only = int(np.mod(time_s, 60))

    time_hr_mn = np.floor(time_s / 60)
    time_mn = int(np.mod(time_hr_mn, 60))
    time_hr = int(np.floor(time_hr_mn / 60))

    print(f"{time_hr}")

    return f"{time_hr:02d}hr:{time_mn:02d}mn:{time_s_only:02d}s"
