import numpy as np
import open3d as o3d
import pybullet as p
from scipy.spatial.transform import Rotation as R


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]]
        )

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic
    
    @classmethod
    def default(cls):
        return cls(width=640, height=480, fx=540, fy=540, cx=320, cy=240)

    def pixel_to_norm_camera_plane(self, uv: np.ndarray) -> np.ndarray:
        xy = (uv - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])
        return xy
    
    def norm_camera_plane_to_pixel(self, xy: np.ndarray, clip=True, round=False) -> np.ndarray:
        uv = xy * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])
        if clip: uv = np.clip(uv, 0, [self.width-1, self.height-1])
        if round: uv = np.round(uv).astype(np.int32)
        return uv


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, intrinsic: CameraIntrinsic, near=0.01, far=4):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.gl_proj_matrix = self.proj_matrix.flatten(order="F")

    def render(self, extrinsic, client=0):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref (^{cam}_{world} T).
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.copy() if extrinsic is not None else np.eye(4)
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")

        result = p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=self.gl_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client
        )

        rgb, z_buffer = np.ascontiguousarray(result[2][:, :, :3]), result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )

        return Frame(rgb, depth, self.intrinsic, extrinsic)
    
    def project(self, extrinsic, points):
        """
        Arguments:
        - extrinsic: (4, 4), cwT
        - points: (N, 3), points in world frame

        Returns:
        - uv: (N, 2), pixel coordinates.
            If points are calculated from the the same frame, then: 
                range of u: [0, W-1];
                range of v: [0, H-1];
        """
        W, H = self.intrinsic.width, self.intrinsic.height
        N = len(points)
        points_homo = np.concatenate([points, np.ones((N, 1))], axis=1)

        proj = self.proj_matrix @ extrinsic @ points_homo.T  # (4, N)
        proj = (proj / proj[-1, :]).T  # (N, 4)
        proj[:, 0] = -proj[:, 0]
        uv = (proj[:, :2] + 1.) * np.array([W, H]) / 2.

        return uv
    
    def filter_occlu(self, wcT, points):

        if isinstance(points, list):
            points = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        camera_location = wcT[:3, 3]
        diameter = np.linalg.norm(np.asarray(pcd.get_min_bound()) - np.asarray(pcd.get_max_bound()))
        visible_pcd, indices = pcd.hidden_point_removal(camera_location, radius=diameter*100)

        return visible_pcd, indices

    def inv_project(self, extrinsic, uv, Z):
        """
        Arguments:
        - extrinsic: (4, 4), ^{cam} _{world} T
        - uv: (N, 2), pixel coordinates,
                range of u: [0, W-1];
                range of v: [0, H-1];
        - Z: (N,), depth in camera frame
        
        Returns:
        - points: (N, 3)
        """
        W, H = self.intrinsic.width, self.intrinsic.height
        N = len(uv)
        inv_proj = uv * 2. / np.array([W, H]) - 1.
        inv_proj[:, 0] = -inv_proj[:, 0]

        f, n = self.far, self.near
        norm_Z = (f+n)/(f-n) + 2*n*f/(f-n) * 1./Z

        inv_proj = np.concatenate([inv_proj, norm_Z[:, None], np.ones((N, 1))], axis=-1)
        inv_proj = inv_proj * -Z[:, None]  # (N, 4)

        X = (inv_proj[:, 0] - Z*self.proj_matrix[0, 2]) / self.proj_matrix[0, 0]
        Y = (inv_proj[:, 1] - Z*self.proj_matrix[1, 2]) / self.proj_matrix[1, 1]
        points = np.stack([X, Y, Z, np.ones_like(Z)], axis=0)  # (4, N)
        points = np.linalg.inv(extrinsic) @ points
        points = points[:3, :].T

        # # equal implementations
        # inv_proj = np.linalg.inv(self.proj_matrix @ extrinsic) @ inv_proj.T  # (4, N)
        # points = inv_proj[:3, :].T
        return points


class Frame(object):
    def __init__(self, rgb, depth, intrinsic, extrinsic=None):
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb),
            depth=o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        self.extrinsic = extrinsic if extrinsic is not None \
            else np.eye(4)
    
    def color_image(self):
        return np.asarray(self.rgbd.color)
    
    def depth_image(self):
        return np.asarray(self.rgbd.depth)

    def point_cloud(self):
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=self.rgbd,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic
        )

        return pc


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho