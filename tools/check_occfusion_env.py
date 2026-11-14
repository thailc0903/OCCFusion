import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mmcv.ops import ModulatedDeformConv2dPack

print("Python:", sys.executable)
assert torch.cuda.is_available(), "CUDA is unavailable"
print("GPU:", torch.cuda.get_device_name(0))

# Kiểm tra DCNv2 trên GPU bằng tensor nhỏ.
layer = ModulatedDeformConv2dPack(
    4, 4, kernel_size=3, padding=1
).cuda().eval()

with torch.no_grad():
    output = layer(torch.randn(1, 4, 16, 16, device="cuda"))

torch.cuda.synchronize()
assert output.shape == (1, 4, 16, 16)
assert torch.isfinite(output).all().item()
print("DCNV2_CUDA: PASS")

# Kiểm tra voxelization; đầu ra dùng thứ tự tọa độ z, y, x.
from mmdet3d.models.data_preprocessors.voxelize import voxelization

points = torch.tensor(
    [
        [0.1, 0.2, 0.3, 1.0],
        [0.4, 0.3, 0.1, 1.0],
        [1.1, 0.6, 1.6, 1.0],
    ],
    dtype=torch.float32,
    device="cuda",
)

coordinates = voxelization(
    points,
    [0.5, 0.5, 0.5],
    [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
    -1,
    -1,
)
torch.cuda.synchronize()

assert coordinates.cpu().tolist() == [
    [0, 0, 0],
    [0, 0, 0],
    [3, 1, 2],
]
print("VOXELIZATION_CUDA: PASS")

# Kiểm tra import và đăng ký model.
import occfusion
from mmengine.config import Config
from mmdet3d.registry import MODELS

print("OccFusion source:", occfusion.__file__)
assert MODELS.get("OccFusion") is not None
print("OCCFUSION_IMPORT: PASS")

cfg = Config.fromfile(str(ROOT / "configs" / "OccFusion.py"))
assert cfg.model.type == "OccFusion"
print("CONFIG_LOAD: PASS")
print("OCCFUSION_ENV: PASS")