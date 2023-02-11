"""
@Date: 2022/9/7
@Description:
open3d is right-hand system
  y
  |
  |_ _ _x
 |
| z
"""
import os

import numpy
import numpy as np
import open3d as o3d


def show_pt(points, colors=None, window_name=''):
    if len(points.shape) > 2:
        points = points.reshape(-1, 3)
    if colors is not None and len(colors.shape) > 2:
        colors = colors.reshape(-1, 3)
    if colors is not None and colors.max() > 1:
        colors = colors / 255

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='vis', width=960, height=640)
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.array([0, 0, 0])


    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points * np.array([[1, -1, -1]]))
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors[..., ::-1])

    vis.add_geometry(point_cloud)
    vis.run()
    vis.destroy_window()


    # o3d.visualization.draw_geometries(
    #     [point_cloud],
    #     window_name=window_name,
    #     zoom=1,
    #     front=[0, 0, -1],
    #     lookat=[0, 0, 0],
    #     up=[0, -1, 0]
    # )


def test():
    points = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [2, 2, 0], [2, 2, 2]])
    show_pt(points)


if __name__ == '__main__':
    test()
