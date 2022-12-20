class BaseAlgorithm:

    image_0_degree = None
    image_180_degree = None

    def __init__(self, index_0_degree=0, index_180_degree=0, proj_mlog=None):
        self.image_0_degree = proj_mlog[index_0_degree]
        self.image_180_degree = proj_mlog[index_180_degree]
