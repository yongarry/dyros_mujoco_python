import mujoco
import mujoco.viewer as mj_view
import time
import threading
import numpy as np
from rl_controller import rl_controller

class mjc_simulator:
    def __init__(self, xml_path, onnx_path):
        # load xml file
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep
        self.ctrl_dof = self.model.nu

        # load onnx file
        # session = ort.InferenceSession(onnx_path)
        # obs = np.zeros((1, 250), dtype=np.float32)
        # onnx_outputs = session.run(None, {"obs": obs})
        # print("Testing outputs from ONNX model:", onnx_outputs)
        self.rc = rl_controller(onnx_path, 0.01, self.dt, self.ctrl_dof)

        self.ctrl_step = 0

        self.running = True
        self.lock = threading.Lock()
        self.robot_ctrl_thread = threading.Thread(target=self.robot_control, daemon=True)

    def run_sim(self):
        scene_update_freq = 30
        try:
            with mj_view.launch_passive(self.model, self.mj_data) as viewer:
                viewer.opt.geomgroup[:] = 0
                viewer.opt.geomgroup[1] = 1
                viewer.opt.geomgroup[3] = 1
                self.robot_ctrl_thread.start()

                while viewer.is_running() and self.running:
                    start_time = time.perf_counter()
                    with self.lock:
                        viewer.sync()
                    self.time_sync(1/scene_update_freq, start_time, False)
        except KeyboardInterrupt:
            print("\nSimulation interrupted. Closing viewer...")
            self.running = False
            self.robot_ctrl_thread.join()

    def time_sync(self, target_dt, t_0, verbose=False):
        elapsed_time = time.perf_counter() - t_0
        sleep_time = target_dt - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        if verbose:
            print(f'Time {elapsed_time*1000:.4f} + {sleep_time*1000:.4f} = {(elapsed_time + sleep_time)*1000} ms')

    def initialize_robot_state(self, initial_state):
        self.mj_data.qpos[:] = initial_state[:]

    def robot_control(self):
        self.ctrl_step = 0

        try:
            while self.running:            
                with self.lock:
                    start_time = time.perf_counter()                        

                    mujoco.mj_step(self.model, self.mj_data)  # 시뮬레이션 실행
                    self.rc.updateModel(self.mj_data, self.ctrl_step)                    
                    self.mj_data.ctrl[:self.ctrl_dof] = self.rc.compute()   

                    self.ctrl_step += 1
                    
                self.time_sync(self.dt, start_time, False)
            
        except KeyboardInterrupt:
            self.get_logger().info("\nSimulation interrupted. Closing robot controller ...")

def main():
    xml_path = "robots/tocabi/dyros_tocabi.xml"
    onnx_path = "policies/tocabi/position/0806-1802_dyrostocabi.onnx"

    sim = mjc_simulator(xml_path, onnx_path)
    initial_state = [   0, 0, 0.92983, 1, 0, 0, 0,
                        0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                        0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                        0, 0, 0,
                        0.3, 0.3, 1.5, -1.27, -1, 0, -1, 0,
                        0, 0,
                        -0.3, -0.3, -1.5, 1.27, 1, 0, 1, 0]
    sim.initialize_robot_state(initial_state)
    sim.run_sim()

if __name__ == "__main__":
    main()