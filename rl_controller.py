import onnxruntime as ort
import numpy as np
import math

class rl_controller:
    def __init__(self, onnx_path, policy_dt, sim_dt, dof):
        # load onnx file
        self.session = ort.InferenceSession(onnx_path)
        self.obs = np.zeros((1, 250), dtype=np.float32)
        onnx_outputs = self.session.run(None, {"obs": self.obs})
        self.action = onnx_outputs[0]
        self.value = onnx_outputs[1]
        print("Testing outputs from ONNX model:", self.action, self.value)
        self.state = np.zeros(50, dtype=np.float32)

        self.time_s = 0.0
        self.policy_dt = policy_dt
        self.sim_dt = sim_dt
        self.sim_step = 0
        self.sim_inference_step = 0

        self.phase_time = 1.2
        self.pd_control = True
        if self.pd_control:
            self.desired_joint_pos = np.zeros(12, dtype=np.float32)

        self.desired_joint_torque = np.zeros(33, dtype=np.float32)

        # simulation data
        self.root_pos = np.zeros(3, dtype=np.float32)
        self.root_quat = np.zeros(4, dtype=np.float32)
        self.root_lin_vel = np.zeros(3, dtype=np.float32)
        self.root_ang_vel = np.zeros(3, dtype=np.float32)
        self.qpos = np.zeros(dof, dtype=np.float32)
        self.qvel = np.zeros(dof, dtype=np.float32)

        # sensor data
        self.lf_ft = np.zeros(6, dtype=np.float32)
        self.rf_ft = np.zeros(6, dtype=np.float32)
        # self.lh_ft = np.zeros(6, dtype=np.float32)
        # self.rh_ft = np.zeros(6, dtype=np.float32)
        # self.acc = np.zeros(3, dtype=np.float32)
        # self.gyro = np.zeros(3, dtype=np.float32)
        # self.magnet = np.zeros(3, dtype=np.float32)
        # self.pelvis_quat = np.zeros(4, dtype=np.float32)
        # self.pelvis_lin_vel = np.zeros(3, dtype=np.float32)
        # self.pelvis_ang_vel = np.zeros(3, dtype=np.float32)

        # joint parameters
        self.default_joint_pos = np.array([   
            0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
            0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
            0, 0, 0,
            0.3, 0.3, 1.5, -1.27, -1, 0, -1, 0,
            0, 0,
            -0.3, -0.3, -1.5, 1.27, 1, 0, 1, 0
        ])
        self.p_gain = np.array([
            2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
            2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
            6000.0, 10000.0, 10000.0,
            400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
            100.0, 100.0,
            400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0
        ])
        self.d_gain = np.array([
            15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
            15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
            200.0, 100.0, 100.0,
            10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
            2.0, 2.0,
            10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0
        ])
        self.action_offset = np.array([
            0.0,  0.0, -0.25,  0.45, -0.15,  0.0,  
            0.0,  0.0, -0.25,  0.45, -0.15,  0.0
        ])
        self.action_scale = np.array([
            0.3, 0.5, 0.75, 0.75, 0.65, 0.6, 
            0.3, 0.5, 0.75, 0.75, 0.65, 0.6,
        ])
        self.joint_effort_limit = np.array([
            333, 232, 263, 289, 222, 166,
            333, 232, 263, 289, 222, 166,
            303, 303, 303, 
            64, 64, 64, 64, 23, 23, 10, 10,
            10, 10,
            64, 64, 64, 64, 23, 23, 10, 10
        ])

    def updateModel(self, data, step):
        self.root_pos = data.qpos[:3]
        self.root_quat = data.qpos[3:7]
        self.root_lin_vel = data.qvel[:3]
        self.root_ang_vel = data.qvel[3:6]

        self.qpos = data.qpos[7:]
        self.qvel = data.qvel[6:]

        self.lf_ft = data.sensordata[:6]
        self.rf_ft = data.sensordata[6:12]
        # self.lh_ft = data.sensordata[12:18]
        # self.rh_ft = data.sensordata[18:24]

        # self.acc = data.sensordata[24:27]
        # self.gyro = data.sensordata[27:30]
        # self.magnet = data.sensordata[30:33]

        # self.pelvis_quat = data.sensordata[33:37]
        # self.pelvis_lin_vel = data.sensordata[37:40]
        # self.pelvis_ang_vel = data.sensordata[40:43]
        
        self.time_s = step * self.sim_dt
        self.sim_step = step
    def compute(self):

        if self.sim_step % 20 == 0:
            self.state = self.compute_observation()
            
            self.obs[:, :-self.state.shape[0]] = self.obs[:, self.state.shape[0]:]
            self.obs[:, -self.state.shape[0]:] = self.state.reshape(1, -1)

            self.action = self.session.run(None, {"obs": self.obs})[0][0]
            self.value = self.session.run(None, {"obs": self.obs})[1][0]


        if self.pd_control:
            self.desired_joint_pos = self.action_offset + self.action_scale * np.clip(self.action, -1.0, 1.0)
            target_joint_pos = np.concatenate((self.desired_joint_pos, self.default_joint_pos[12:]))
            self.desired_joint_torque = self.p_gain / 9.0 * (target_joint_pos - self.qpos) - self.d_gain / 3.0 * self.qvel
        else:
            lower_body_torque = np.clip(self.action, -1.0, 1.0) * self.joint_effort_limit[:12]
            upper_body_torque = self.p_gain[12:] / 9.0 * (self.default_joint_pos[12:] - self.qpos[12:]) - self.d_gain[12:] / 3.0 * self.qvel[12:]
            self.desired_joint_torque = np.concatenate((lower_body_torque, upper_body_torque))
        
        return self.desired_joint_torque

    def compute_observation(self):
        state = np.zeros(self.state.shape, dtype=np.float32)
        # 1. lin vel , ang vel on local base frame
        state[0:3] = quatRotateInverse(self.root_quat, self.root_lin_vel)
        state[3:6] = quatRotateInverse(self.root_quat, self.root_ang_vel)
        
        # 2. projected gravity vector
        state[6:9] = quatRotateInverse(self.root_quat, np.array([0.0, 0.0, -1.0]))

        # 3. clock input
        state[9:11] = np.concatenate((
            np.array([math.sin(self.time_s * 2.0 * math.pi / self.phase_time)]),
            np.array([-math.sin(self.time_s * 2.0 * math.pi / self.phase_time)])
        ))

        # 4. base velocity cmd
        state[11] = 0.3
        state[12] = 0.0
        state[13] = 0.0

        # 5. joint position
        state[14:26] = self.qpos[:12] - self.default_joint_pos[:12]

        # 6. joint velocity
        state[26:38] = self.qvel[:12]

        # 7. last action
        if self.pd_control:
            state[38:] = self.desired_joint_pos
        else:
            state[38:] = np.clip(self.action, -1.0, 1.0)

        return state


def quatRotateInverse(quat, vec):
    '''
    quat : quaternion (w, x, y, z)
    vec : vector (x, y, z)
    '''
    q_vec = quat[1:]
    a = vec * (2.0 * quat[0] ** 2 - 1.0)
    b = 2.0 * quat[0] * np.cross(q_vec, vec)
    c = 2.0 * q_vec * np.dot(q_vec, vec)

    return a - b + c
