import numpy as np

from mpsfm.utils.geometry import project3D, unproject_depth_map_to_world


class DepthUtils:
    """Depth map utils mixin for reconstruction."""

    def reproject_depth(self, imid1, imid2, cfw1=None, cfw2=None):
        """Reprojects depth from imid1 to imid2"""
        image1 = self.images[imid1]
        image2 = self.images[imid2]
        depth1 = self.images[imid1].depth.data
        depth2_shape = self.images[imid2].depth.data_prior.shape

        cam1 = self.cameras[self.images[imid1].camera_id]
        cam2 = self.cameras[self.images[imid2].camera_id]
        valid1_mask = np.ones(depth1.shape, dtype=bool)
        depth1[depth1 <= 0] = 0.1
        if cfw1 is None:
            cfw1 = image1.cam_from_world
        if cfw2 is None:
            cfw2 = image2.cam_from_world
        H1 = np.concatenate([cfw1.matrix(), np.array([[0, 0, 0, 1]])], axis=0)
        H2 = np.concatenate([cfw2.matrix(), np.array([[0, 0, 0, 1]])], axis=0)
        K1 = cam1.calibration_matrix()
        K1[0, :] *= cam1.sx
        K1[1, :] *= cam1.sy
        D1W = unproject_depth_map_to_world(depth1, K1, np.linalg.inv(H1), mask=valid1_mask)
        K2 = cam2.calibration_matrix()
        K2[0, :] *= cam2.sx
        K2[1, :] *= cam2.sy
        p2D12, depth12 = project3D(D1W, H2, K2)

        mask12 = (
            (p2D12[:, 0] >= 0)
            & ((p2D12[:, 0] + 0.5) < depth2_shape[1])
            & (p2D12[:, 1] >= 0)
            & ((p2D12[:, 1] + 0.5) < depth2_shape[0])
        )
        mask12 *= depth12 > 0
        out = {
            "depth1": depth1,
            "p2D12": p2D12.reshape(*(depth1.shape), 2),
            "depth12": depth12.reshape(depth1.shape),
            "mask12": mask12.reshape(depth1.shape),
            "valid1_mask": valid1_mask,
        }

        return out

    def activate_depths(self, imids):
        """Activates depths for images"""
        for imid in imids:
            if not self.images[imid].depth.activated:
                self.images[imid].depth.activated = True
                self.images[imid].depth.data = self.images[imid].depth.data_prior.copy()

    def _rescale_prior(self, imid, shift, scale):
        """Rescales depth prior"""
        self.images[imid].depth.data_prior = self.images[imid].depth.data_prior * scale + shift
        self.images[imid].depth.scale *= scale
        self.images[imid].depth.shift = self.images[imid].depth.shift * scale + shift
        self.images[imid].depth.uncertainty *= scale**2

    def __rescale_update(self, imid, shift, scale, rescale_depth=False):
        """Rescales optimized depth"""
        if rescale_depth and self.images[imid].depth.activated:
            self.images[imid].depth.data = self.images[imid].depth.data * scale + shift

        self.images[imid].depth.uncertainty_update *= scale**2

    def rescale_all(self, shift_scales):
        """Rescales all depths i.e. priors and optimized depths"""
        self.rescale_priors(shift_scales)
        self.rescale_depths(shift_scales)

    def rescale_depths(self, shift_scales):
        """Rescales optimized depths"""
        for imid, (shift, scale) in shift_scales.items():
            self.__rescale_update(imid, shift, scale)

    def rescale_priors(self, shift_scales):
        """Rescales priors"""
        for imid, (shift, scale) in shift_scales.items():
            self._rescale_prior(imid, shift, scale)

    def normalize_depths(self, scale):
        """Normalizes depths"""
        for imid in self.images:
            self._rescale_prior(imid, 0, scale)
            self.__rescale_update(imid, 0, scale, rescale_depth=True)
