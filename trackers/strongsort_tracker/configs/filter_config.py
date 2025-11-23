
class Exp(object):
    def __init__(self,
                 min_confidence=0.8,
                 max_iou_distance=0.7,
                 max_age=30,
                 n_init=3,
                 woC=True,
                 NSA=True,
                 EMA=True,
                 EMA_alpha=0.9,
                 MC=True,
                 MC_lambda=0.98,
                 max_dist=0.2,
                 category_aware_tracking=True,
                 ):

        local_vars = locals().copy()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        self.with_reid = True
        self.with_gmc = True
        # self.cmc_method = 'orb'
        self.gmc_method = 'ecc'
        self.gmc_downscale = 4
        self.tracker_name = "StrongSORT"

#
# def get_gaussian_density_NuScenes_CV() -> GaussianDensity:
#     """
#     Set up motion & measurement model for NuScenes, Constant Velocity
#     :return:
#     """
#     state_dim = 6   # [x, y, yaw, vx, vy, vyaw]
#     meas_dim = 3  # [x, y, yaw]
#     dt = 0.5  # sampling time
#     # motion model
#     F = np.eye(state_dim)
#     F[:3, 3:] = np.eye(3) * dt
#     # Q = np.diag([2.0, 2.0, 0.5, 5.0, 5.0, 1.0])
#     Q = {
#         'bicycle':  np.diag([1.98881347e-02, 1.36552276e-02, 1.33430252e-01, 1.98881347e-02, 1.36552276e-02, 1.33430252e-01]),
#         'bus':      np.diag([1.17729925e-01, 8.84659079e-02, 2.09050032e-01, 1.17729925e-01, 8.84659079e-02, 2.09050032e-01]),
#         'car':      np.diag([1.58918523e-01, 1.24935318e-01, 9.22800791e-02, 1.58918523e-01, 1.24935318e-01, 9.22800791e-02]),
#         'motorcycle': np.diag([3.23647590e-02, 3.86650974e-02, 2.34967407e-01, 3.23647590e-02, 3.86650974e-02, 2.34967407e-01]),
#         'pedestrian': np.diag([3.34814566e-02, 2.47354921e-02, 4.24962535e-01, 3.34814566e-02, 2.47354921e-02, 4.24962535e-01]),
#         'trailer':  np.diag([4.19985099e-02, 3.68661552e-02, 5.63166240e-02, 4.19985099e-02, 3.68661552e-02, 5.63166240e-02]),
#         'truck':    np.diag([9.45275998e-02, 9.45620374e-02, 1.41680460e-01, 9.45275998e-02, 9.45620374e-02, 1.41680460e-01])
#     }
#     # measurement model
#     H = np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)
#     # R = np.diag([0.65, 0.65, 1.0])
#     R = {
#         'bicycle':  np.diag([0.05390982, 0.05039431, 1.29464435]),
#         'bus':      np.diag([0.17546469, 0.13818929, 0.1979503]),
#         'car':      np.diag([0.08900372, 0.09412005, 1.00535696]),
#         'motorcycle': np.diag([0.04052819, 0.0398904, 1.06442726]),
#         'pedestrian': np.diag([0.03855275, 0.0377111, 2.0751833]),
#         'trailer':  np.diag([0.23228021, 0.22229261, 1.05163481]),
#         'truck':    np.diag([0.14862173, 0.1444596, 0.73122169])
#     }
#     density = GaussianDensity(state_dim, meas_dim, F, Q, H, R)
#     return density
