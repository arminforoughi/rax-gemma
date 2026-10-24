"""The Luxonis OAK-D as a LeRobot camera, for a stock ``pip install lerobot``.

Stock LeRobot has no OAK-D camera; this is the one from the RAX LeRobot fork, with its
imports pointed at the installed ``lerobot.cameras``. Importing it registers the
``"oakd"`` camera type, and LeRobot's camera factory finds :class:`OAKDCamera` here by
name, so nothing in LeRobot is patched. If the fork is installed, ``rax.robots.so101``
uses the fork's copy instead and this one is never imported (the type can only be
registered once).
"""

from .camera_oakd import OAKDCamera
from .configuration_oakd import OAKDCameraConfig

__all__ = ["OAKDCamera", "OAKDCameraConfig"]
