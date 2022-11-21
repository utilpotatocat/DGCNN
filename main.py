import numpy as np
import open3d as o3d


def draw_pointcloud(points,seg):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])



points = np.random.rand(10000, 3)
print(points)
print(points.shape)
seg=[]

draw_pointcloud(points,seg)
