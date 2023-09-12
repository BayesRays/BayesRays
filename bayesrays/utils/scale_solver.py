import struct
import numpy as np
import torch
import argparse
from pathlib import Path
from scipy.optimize import least_squares

#parts of code from https://github.com/dunbar12138/DSNeRF/blob/main/colmapUtils


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points_bin(base_dir):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points_dir = Path('points3D.bin')
    final_dir = base_dir / points_dir
    points3D = {}
    points3D_loc = {}
    xyzs = []
    with open(final_dir, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = {
                'id':point3D_id, 'xyz':xyz, 'rgb':rgb,
                'error':error, 'image_ids':image_ids,
                'point2D_idxs':point2D_idxs}
            points3D_loc[point3D_id] = xyz
    return points3D, points3D_loc      


def read_images_bin(base_dir):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images_dir = Path('images.bin')
    final_dir = base_dir / images_dir
    images = {}
    with open(final_dir, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = {
                'id':image_id, 'qvec':qvec, 'tvec':tvec,
                'camera_id':camera_id, 'name':image_name,
                'xys':xys, 'point3D_ids':point3D_ids}
    return images       

def quaternion_to_rotation_matrix(qvec):
    qw, qx, qy, qz = qvec
    
    # Compute terms for clarity
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qwqx = qw * qx
    qwqy = qw * qy
    qwqz = qw * qz
    qxqy = qx * qy
    qxqz = qx * qz
    qyqz = qy * qz
    
    # Compute rotation matrix
    R = np.array([[1 - 2*qy2 - 2*qz2, 2*qxqy - 2*qz*qw, 2*qxqz + 2*qy*qw],
                  [2*qxqy + 2*qz*qw, 1 - 2*qx2 - 2*qz2, 2*qyqz - 2*qx*qw],
                  [2*qxqz - 2*qy*qw, 2*qyqz + 2*qx*qw, 1 - 2*qx2 - 2*qy2]])
    
    return R

def find_depths(images, points,scale_factor, downscale, auto_scaled):
    camera_centers = []
    Rs = []
    depths_all = []
    xys_all = []
    scale = 1
    for i in range(len(images)):     
        R = quaternion_to_rotation_matrix(images[i+1]['qvec'])
        Rs += [R.T]
        camera_center = -R.T @ images[i+1]['tvec']
        camera_centers += [list(camera_center)]
    camera_centers = np.array(camera_centers)
    if auto_scaled:
        scale *= 1/ float(np.max(np.abs(camera_centers)))
    scale *= scale_factor
        
    for i in i_val:
        xys = np.flip(images[i+1]['xys'],-1) #flip the order of x and y (result: x indexes rows, and y columns)
        xys = xys [images[i+1]['point3D_ids'] != -1] #mask out non 3D keypoints
        
        xys = xys / downscale
        camera_center = camera_centers[i]
        points3D = np.array([list(points[j]) for j in images[i+1]['point3D_ids'] if j!=-1])
        depths =  (points3D - camera_center) @ Rs[i][:3,2]
        depths = depths * scale 
        depths_all += [depths]
        xys_all += [xys]
    
    return xys_all, depths_all #num_images ( point of image x 3|1)

def read_gt_depth(base_dir, xys):
    depths = []
    for c,i in enumerate(i_val):
        depth_gt_dir =  base_dir / Path('depth_gt_{:02d}.npy'.format(c))
        depth_gt = np.load(str(depth_gt_dir))
        xs = np.floor(xys[c][:, 0]).astype(int)
        ys = np.floor(xys[c][:, 1]).astype(int)
        depth_gt_points = depth_gt[xs, ys]
        depths+=[depth_gt_points]
    return depths   #num_images ( point of image x 1)




def main(args):
    
    print(args)
    base_dir = Path('./data') / Path(args.dataset) / Path(args.scene) 
    colmap_dir = base_dir / Path('colmap/sparse/0')
    _, points = read_points_bin(colmap_dir)
    images = read_images_bin(colmap_dir)
    xys, colmap_depths = find_depths(images, points, args.scale_factor, args.downscale, args.autoscale)
    gt_depths = read_gt_depth(base_dir, xys)
    
    colmap_depths = np.concatenate(colmap_depths)
    gt_depths = np.concatenate(gt_depths)
    
    
    def objective(x):
        a = x
        residual = a * colmap_depths - gt_depths
        return residual

    # Initial guess for parameters
    x0 = [1.0]

    # Solve the least squares problem
    result = least_squares(objective, x0)
    
    # Extract the optimal parameters
    a_opt = result.x
    a = a_opt

    print("Optimal parameters:")
    print("a =", a_opt)
    print(np.mean(colmap_depths), np.max(colmap_depths), np.min(colmap_depths), np.median(colmap_depths))
    print(np.mean(gt_depths), np.max(gt_depths), np.min(gt_depths),np.median(gt_depths))
    print(np.mean(a * colmap_depths ), np.max(a * colmap_depths ), np.min(a * colmap_depths), np.median(gt_depths)/np.median(colmap_depths))
    
    
    #save a and b
    save_dir = base_dir / Path('scale_parameters.txt')
    np.savetxt(str(save_dir), [a_opt], delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your script description')
    
    # Add arguments to the parser
    parser.add_argument('--dataset', type=str, default='scannet', help='dataset name')
    parser.add_argument('--scene', type=str, default='scene_079', help='scene name')
    parser.add_argument('--scale-factor', type=float, default=1., help='scale factor')
    parser.add_argument('--downscale', type=float, default=2., help='image downscale')
    parser.add_argument("--autoscale", action="store_const", const=True, default=False, help="auto scaled")
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.dataset == 'scannet':
        i_val = [4, 12, 20, 28, 36] 
    elif args.dataset == 'LF':
        if args.scene == 'basket':
            i_val = list(np.arange(42,50,2))
        elif args.scene == 'africa':
            i_val = list(np.arange(6,14,2))
        elif args.scene == "torch":
            i_val = list(np.arange(9,17,2))
        elif args.scene == "statue":
            i_val = list(np.arange(68,76,2))
        
    
    main(args)
