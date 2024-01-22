# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np

from aitviewer.renderables.rigid_bodies import RigidBodies


class CoordinateSystem(RigidBodies):
    """
    Render a coordinate system using shaded cylinders.
    """

    def __init__(self, length=1.0, icon="\u008a", rb_pos = np.array([[[0.0, 0.0, 0.0]]]), rb_ori = np.eye(3)[np.newaxis, np.newaxis], **kwargs):
        r = length / 50
        l = length
        super(CoordinateSystem, self).__init__(
            rb_pos,
            rb_ori,
            radius=r,
            length=l,
            icon=icon,
            **kwargs,
        )