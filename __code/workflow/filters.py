import timeit

try:
    from imars3d.backend.corrections.ring_removal import remove_ring_artifact
    enable_remove_ring_artifact = True
except OSError:
    enable_remove_ring_artifact = False

from __code.parent import Parent
from __code import NCORE


class Filters(Parent):

    def remove_negative_values(self):
        """remove all the intensity that are below 0"""
        if self.parent.remove_negative_ui.value:
            self.parent.proj_mlog[self.parent.proj_mlog < 0] = 0
            print(" Removed negative values!")
        else:
            print(" Skipped remove negative values!")

    def strikes_removal(self):
        if self.parent.strikes_removal_ui.value:
            t0 = timeit.default_timer()
            print("Running strikes removal ...")
            self.parent.proj_strikes_removed = remove_ring_artifact(arrays=self.parent.proj_tilt_corrected,
                                                                    kernel_size=5,
                                                                    max_workers=NCORE)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            self.parent.proj_strikes_removed = self.parent.proj_tilt_corrected
            print(" Skipped strikes removal!")
