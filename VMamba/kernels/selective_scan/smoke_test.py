import os

os.add_dll_directory(r"E:\anaconda3\envs\landslide\bin")
os.add_dll_directory(r"E:\anaconda3\envs\landslide\Lib\site-packages\torch\lib")

import torch
from test_selective_scan import test_selective_scan

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

test_selective_scan(
    True,              # is_variable_B
    True,              # is_variable_C
    2,                 # varBC_groups
    True,              # has_D
    False,             # has_z
    True,              # has_delta_bias
    True,              # delta_softplus
    True,              # return_last_state
    4096,              # seqlen
    torch.float16,     # itype
    torch.float32,     # wtype
    4,                 # nrows
    2,                 # batch_size
    768,               # dim
    24,                # dim1
    1                  # dstate
)