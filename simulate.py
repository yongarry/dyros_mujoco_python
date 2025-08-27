import mujoco
import mujoco.viewer as mj_view
import time
import onnxruntime as ort
import numpy as np

# load xml file
xml_path = "robots/tocabi/dyros_tocabi.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# load onnx file 
session = ort.InferenceSession("policies/tocabi/position/0806-1802_dyrostocabi.onnx")
obs = np.zeros((1, 250), dtype=np.float32)
action = session.run(None, {"obs": obs})[0]
print("Action from ONNX model:", action)

# initialize root state of the model
data.qpos[:] = [0, 0, 0.92983, 1, 0, 0, 0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0, 0, 0,
                0.3, 0.3, 1.5, -1.27, -1, 0, -1, 0,
                0, 0,
                -0.3, -0.3, -1.5, 1.27, 1, 0, 1, 0]

try:
    with mj_view.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[3] = 1

        while viewer.is_running():
            mujoco.mj_step(model, data)  
            viewer.sync()  
            time.sleep(0.001) 

except KeyboardInterrupt:
    print("\nSimulation interrupted. Closing viewer...")
