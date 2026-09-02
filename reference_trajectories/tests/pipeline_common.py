# TEST STUB ONLY - the real pipeline_common.py lives next to the other stages
import numpy as np
from scipy.spatial.transform import Rotation as Rot
def R_to_q(R):
    return Rot.from_matrix(R).as_quat()
def load_pipeline(path):
    raise NotImplementedError("stub")
