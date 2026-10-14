import os
import pathlib
import sys

# The tests run on the CPU only. Hide the GPU before torch is imported, so no test can use it.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
