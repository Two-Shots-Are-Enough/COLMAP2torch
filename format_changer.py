# Copyright (c) yoomimi
'''
format_changer.py is a script that converts colmap format to .torch format (re10k method).
It reads intrinsic and extrinsic parameters and saves them as .torch files.
Although there are 16 meaningful values, cameras contain 18 values because
the convert_poses function in dataset_re10k.py uses poses[:, :4] and poses[:, 6:],
and the .torch cameras in re10k are composed of 18 values.
Therefore, we forcefully insert 0s at [4:6] to match the format.
'''


from pathlib import Path
import torch
import json
import numpy as np
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from PIL import Image
from torch import Tensor
from read_write_model import Camera, Image as ColmapImage, read_model
import hashlib

def read_colmap_model(
    path: Path,
    device: torch.device = torch.device("cpu"),
    ext=".bin"  # 명시적으로 파일 형식 bin 지정. txt로 하려면 txt로 변경(한다고 되는지는 체크해봐야 함. 일단 bin으로.).
) -> tuple[
    Float[Tensor, "frame 4 4"],  # extrinsics
    Float[Tensor, "frame 3 3"],  # intrinsics
    list[str],  # scene names
]:
    cameras, images, _ = read_model(path, ext=ext)  # 파일 형식 전달

    if cameras is None or images is None:
        raise FileNotFoundError(f"COLMAP model not found at {path} with format {ext}")

    all_extrinsics = []
    all_intrinsics = []
    all_image_names = []

    for image in images.values():
        camera: Camera = cameras[image.camera_id]

        # Read the camera intrinsics.
        intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        if camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        elif camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        intrinsics[0, 0] = fx / camera.width  # normalize to image width (for fitting to re10k design)
        intrinsics[1, 1] = fy / camera.height  # normalize to image height (for fitting to re10k design)
        intrinsics[0, 2] = cx / camera.width
        intrinsics[1, 2] = cy / camera.height
        all_intrinsics.append(intrinsics)

        # Read the camera extrinsics.
        # mipnerf360의 경우 `c2w`를 읽어옴.
        qw, qx, qy, qz = image.qvec
        c2w = torch.eye(4, dtype=torch.float32, device=device)  # Camera-to-World
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
        c2w[:3, 3] = torch.tensor(image.tvec, dtype=torch.float32, device=device)
        all_extrinsics.append(c2w)

        # Read the image name.
        all_image_names.append(image.name)

    return torch.stack(all_extrinsics), torch.stack(all_intrinsics), all_image_names


def colmap_to_pixelsplat_all(
    colmap_base_path: Path, output_path: Path, device: torch.device = torch.device("cpu")
) -> None:
    """
    Convert multiple COLMAP format datasets to a Pixelsplat format (.torch files and index.json),
    including RGB image data, camera parameters, and scene index keys.
    """
    
    output_path.mkdir(parents=True, exist_ok=True)
    index_mapping = {}
    torch_file_idx = 0  # Global index for .torch filenames
    
    for dataset_path in colmap_base_path.iterdir():
        if not dataset_path.is_dir():
            continue

        sparse_path = dataset_path / "sparse/0"
        image_path = dataset_path / "images"
        if not sparse_path.exists() or not image_path.exists():
            print(f"No sparse or image data found in {dataset_path}, skipping.")
            continue

        try:
            extrinsics, intrinsics, image_names = read_colmap_model(sparse_path, device=device)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Proceeding without points3D data.")
            continue

        num_images = extrinsics.shape[0]
        camera_data = torch.zeros((num_images, 18), dtype=torch.float32)
        image_data = []
        scene_key = hashlib.md5(dataset_path.name.encode()).hexdigest()[:16]  # Generate a scene index key
        
        print(f"Dataset '{dataset_path.name}' -> Scene Key: {scene_key}")

        # Iterate over each image's intrinsics and extrinsics
        for idx, (c2w, K, img_name) in enumerate(zip(extrinsics, intrinsics, image_names)):
            img_file = image_path / img_name
            if not img_file.exists():
                print(f"Image file {img_file} does not exist, skipping.")
                continue
            
            # rgb flatten 아니고 그냥 byte로 읽어오기
            with open(img_file, 'rb') as f:
                image_bytes = f.read()  # Read image as raw bytes

            # Convert raw bytes to tensor
            image_data.append(torch.tensor(list(image_bytes), dtype=torch.uint8))

            
            # Extract normalized intrinsics
            fx, fy = K[0, 0].item(), K[1, 1].item()
            cx, cy = K[0, 2].item(), K[1, 2].item()
            
            # MVSplat은 w2c를 받음. 따라서 c2w를 w2c로 변환해야 함.
            w2c = np.linalg.inv(c2w.cpu().numpy()) # 4x4 matrix
            extrinsic_matrix = w2c[:3,:].flatten()  # flatten 3x4 matrix to 12 values

            # Set camera_data for current image with proper alignment
            camera_data[idx, :4] = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)  # Intrinsics
            camera_data[idx, 4:6] = torch.tensor([0, 0], dtype=torch.float32)  # Padding
            camera_data[idx, 6:18] = torch.tensor(extrinsic_matrix, dtype=torch.float32) 

        # Save to .torch file
        torch_file = output_path / f"{str(torch_file_idx).zfill(6)}.torch"
        torch.save([  # format 엄밀히 맞추기 위해 List로 감싸서 저장해야 함.
            {
                'cameras': camera_data,
                'images': image_data,
                'key': scene_key
            }
        ], torch_file)

        # Map the scene_key to the torch file in index.json
        index_mapping[scene_key] = torch_file.name
        torch_file_idx += 1

    # Write index.json
    with open(output_path / "index.json", 'w') as f:
        json.dump(index_mapping, f, indent=4)

# Usage example:
colmap_base_path = Path("mipnerf360")  # Base path containing multiple datasets (e.g., bicycle, bonsai)
output_path = Path("mipnerf/test")          # Output directory for Pixelsplat format
colmap_to_pixelsplat_all(colmap_base_path, output_path)
