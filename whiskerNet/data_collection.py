import numpy as np
import matplotlib.pyplot as plt
import time
from scripts.utils import generate_path, get_contact_force, get_whisker_base_force_base_frame
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import hydra
from dm_control import mjcf, mujoco
import mujoco.viewer 
import random
import os

def get_cad_parts_from_path(path, parts_dir="parts"):
    paths = os.listdir(os.path.join(path, parts_dir))
    idx_after_models = path.find("models") + len("models") + 1
    paths = [os.path.join(path[idx_after_models:], parts_dir, p) for p in paths]
    return paths
    

def get_random_rot():
    # generate random quaternion
    u = np.random.rand(3)
    u = u / np.linalg.norm(u)
    theta = 2 * np.pi * np.random.rand()
    q = np.array([
        np.cos(theta / 2),
        u[0] * np.sin(theta / 2), u[1] * np.sin(theta / 2), u[2] * np.sin(theta / 2)])
    rot = np.zeros((9, 1))
    mujoco.mju_quat2Mat(rot, q)

    return rot.reshape(3,3), q

class RawSensorData():
    def __init__(self, torque_scale=1e7):
        self.data = {
            "force": [],
            "torque": [],
            "contact_pos": [],
            "whisker_base_pos": [],
            "whisker_shapes": [],
            "object_pose": [],
            "whisker_vel": []
        }
        self.torque_scale = torque_scale
        self.start_idx = 0
        self.end_idx = 0
        self.torque_too_high = False
        self.skipped = False
    
    def append(self, force, torque, contact_pos, whisker_base_pos, whisker_shapes):
        self.data["force"].append(force)
        self.data["torque"].append(torque)
        self.data["contact_pos"].append(contact_pos)
        self.data["whisker_base_pos"].append(whisker_base_pos)
        self.data["whisker_shapes"].append(whisker_shapes)

    def find_first_last_valid(self):
        # find the first non-zero contract_pos
        for idx, contact_pos in enumerate(self.data["contact_pos"]):
            if np.linalg.norm(contact_pos) > 0:
                self.start_idx = idx
                break
        
        # find the last non-zero contract_pos
        for idx, contact_pos in enumerate(reversed(self.data["contact_pos"])):
            if np.linalg.norm(contact_pos) > 0:
                self.end_idx = len(self.data["contact_pos"]) - idx
                break
    def plot(self):
        # plot the force and torque
        force = np.array(self.data["force"])
        torque = np.array(self.data["torque"]) * self.torque_scale
        contact_pos = np.array(self.data["contact_pos"])

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].plot(force)
        ax[0, 0].set_title("Torque x")
        # print(force.shape, torque.shape)
        ax[0, 0].scatter(np.linspace(0, torque.shape[0], torque.shape[0]), torque[:,0], s=1)
        ax[0, 1].set_title("Torque y")
        ax[0, 1].scatter(np.linspace(0, torque.shape[0], torque.shape[0]), torque[:,1], s=1)
        ax[1, 0].set_title("Torque z")
        ax[1, 0].scatter(np.linspace(0, torque.shape[0], torque.shape[0]), torque[:,2], s=1)
        ax[1, 1].set_title("Contact pos")
        ax[1, 1].plot(contact_pos)

        plt.show()
    def export_to_csv(self, save_force, save_torque, save_path, object_name, traj_idx, sample_pts, keep_intro_outro_0=150):
        save_path = save_path.replace("timestamp", time.strftime("%Y%m%d-%H%M%S"))
        save_path = save_path.replace("objectname", object_name+"_"+str(traj_idx))

        # find the first and last non-zero contact_pos
        self.find_first_last_valid()
        start_idx_k = self.start_idx-keep_intro_outro_0 if self.start_idx > keep_intro_outro_0 else 0
        end_idx_k = self.end_idx+keep_intro_outro_0 if self.end_idx < len(self.data["force"]) - keep_intro_outro_0 else len(self.data["force"])

        self.data["force"] = np.array(self.data["force"][start_idx_k:end_idx_k])
        self.data["torque"] = np.array(self.data["torque"][start_idx_k:end_idx_k])
        self.data["contact_pos"] = np.array(self.data["contact_pos"][start_idx_k:end_idx_k])
        self.data["whisker_base_pos"] = np.array(self.data["whisker_base_pos"][start_idx_k:end_idx_k])
        # self.data["whisker_shapes"] = np.array(self.data["whisker_shapes"][start_idx_k:end_idx_k])

        # sample the data
        if sample_pts > 0:
            data_length = 20000 
            if len(self.data["force"]) < data_length:
                # fill in 0 to make the length 20000
                self.data["force"] = np.concatenate([
                    self.data["force"], np.zeros((data_length - len(self.data["force"]), self.data["force"].shape[1]))], axis=0)
                self.data["torque"] = np.concatenate([
                    self.data["torque"], np.zeros((data_length - len(self.data["torque"]), self.data["torque"].shape[1]))], axis=0)
                self.data["contact_pos"] = np.concatenate([
                    self.data["contact_pos"], np.zeros((data_length - len(self.data["contact_pos"]), self.data["contact_pos"].shape[1]))], axis=0)
                self.data["whisker_base_pos"] = np.concatenate([
                    self.data["whisker_base_pos"], np.zeros((data_length - len(self.data["whisker_base_pos"]), self.data["whisker_base_pos"].shape[1]))], axis=0)
                
            else:
                # cut the data to 20000
                self.data["force"] = self.data["force"][:data_length]
                self.data["torque"] = self.data["torque"][:data_length]
                self.data["contact_pos"] = self.data["contact_pos"][:data_length]
                self.data["whisker_base_pos"] = self.data["whisker_base_pos"][:data_length]

            sample_idx = np.linspace(0, len(self.data["force"]) - 1, sample_pts, dtype=int)
            self.data["force"] = np.array(self.data["force"])[sample_idx]
            self.data["torque"] = np.array(self.data["torque"])[sample_idx]
            self.data["contact_pos"] = np.array(self.data["contact_pos"])[sample_idx]
            self.data["whisker_base_pos"] = np.array(self.data["whisker_base_pos"])[sample_idx]
        
        if save_force:
            if self.torque_too_high or self.skipped:
                print(f"Skip saving force {object_name} {traj_idx}")
                return
            # Save the results
            f_results_all = np.hstack([
                self.data["contact_pos"], self.data["force"], self.data["whisker_base_pos"]])
            f_save_path = save_path[:-5]+ "_force.csv"

            df = pd.DataFrame(f_results_all, columns=['cx', 'cy', 'cz', 'sx', 'sy', 'sz', "wbx", "wby", "wbz"])
            df.to_csv(f_save_path, index=True)
            print(f"Saved to {f_save_path}")
        if save_torque:
            if self.torque_too_high or self.skipped:
                print(f"Skip saving torque {object_name} {traj_idx}")
                return
            # Save the results
            t_results_all = np.hstack([
                self.data["contact_pos"], np.array(self.data["torque"]) * self.torque_scale, 
                self.data["whisker_base_pos"]])
            t_save_path = save_path[:-5]+ "_torque.csv"
            # add title px, py, pz, sx, sy, sz
            cols = ['cx', 'cy', 'cz', 'sx', 'sy', 'sz', "wbx", "wby", "wbz"]
            
            df = pd.DataFrame(t_results_all, columns=cols)
            df.to_csv(t_save_path, index=True)
            print(f"Saved to {t_save_path}")
    
    

@hydra.main(version_base=None, config_path="./config", config_name="data_collect")
def main(cfg):
    env_config = cfg.environment

    # Load your MuJoCo XML model and create a simulation
    
    asset_path = os.path.join("../mujoco_xml/asset/", env_config.model_folder)

    object_names = [folder for folder in os.listdir(asset_path) if os.path.isdir(os.path.join(asset_path, folder))]
    object_names = sorted(object_names)

    invalid_traj = 0
    for object_name in object_names[env_config.start_idx:]:
        print(f"Processing {object_name}")
        with open(env_config.xml_file, "r") as xml_file:
            mjcf_model = mjcf.from_file(xml_file)
        mjcf_model.compiler.meshdir = asset_path

        # for object_name in sampled_objects:
        object_dir = os.path.join(object_name, "google_16k")
        mesh = mjcf_model.asset.add(
            'mesh', name=object_name, file=os.path.join(object_dir, "nontextured.stl"), 
            )

        # add the object to the body
        body = mjcf_model.worldbody.add('body', name=object_name)
        body.add('geom', **{"mesh": mesh, "class": "visual", "name": object_name + "_geom"})

        # add the collision object
        paths = get_cad_parts_from_path(os.path.join(asset_path, object_dir))
        for i, path in enumerate(paths):
            mesh = mjcf_model.asset.add('mesh', name=object_name + str(i), file=path)
            body.add('geom', **{"mesh": mesh, "class": "collision", "name": object_name + str(i) + "_geom"})

        # build the model
        models = [mujoco.MjModel.from_xml_string(mjcf_model.to_xml_string(), mjcf_model.get_assets()) for _ in range(env_config.num_envs)] 
        datas = [mujoco.MjData(m) for m in models]
        print("Model built")

        max_xs = []
        for traj_idx in tqdm(range(env_config.traj_per_obj_env)):
            try:
                skipping_index = [False] * env_config.num_envs
                for i, (m, d) in enumerate(zip(models, datas)):
                    m.opt.gravity = (0,0,0)
                    m.opt.timestep = 0.001
                    mujoco.mj_resetData(m, d)

                    # start random geometry
                    print(f"Start simulation env {i}")
                    obj_geom = m.mesh(object_name)

                    vert_start, vert_end = obj_geom.vertadr[0], obj_geom.vertadr[0] + obj_geom.vertnum[0]
                    obj_verts = m.mesh_vert[vert_start:vert_end, :]

                    mesh_pos = m.mesh_pos[obj_geom.id]
                    mesh_quat = m.mesh_quat[obj_geom.id]
                    
                    trans_mat = np.zeros((9, 1))
                    mujoco.mju_quat2Mat(trans_mat, mesh_quat)
                    transformed_verts = (
                        trans_mat.reshape(3,3) @ obj_verts.T).T + mesh_pos

                    rot_mat, quat = get_random_rot()
                    obj_body = m.body(object_name)
                    obj_body.quat = quat

                    transformed_verts = (rot_mat @ transformed_verts.T).T

                    min_z, max_z = np.min(transformed_verts[:, 2]), np.max(transformed_verts[:, 2])
                    sensor_z = random.uniform(min_z + (max_z-min_z) * 0.1, max_z - (max_z-min_z) * 0.1)

                    eps_z = 10e-4
                    select_pts = transformed_verts[np.abs(transformed_verts[:, 2] - sensor_z) < eps_z]
                    if len(select_pts) < 20:
                        print(f"Skip simulation env {i}")
                        skipping_index[i] = True
                        finished_envs[i] = True
                        continue

                    min_x, min_y = np.min(select_pts[:, 0]), np.min(select_pts[:, 1])

                    sensing_range = [0.01, 0.038]
                    
                    target_y = random.uniform(sensing_range[0], sensing_range[1])
                    delta_y = target_y - min_y

                    target_x = 0.010
                    delta_x = target_x - min_x

                    target_z = 0.05
                    delta_z = target_z - sensor_z

                    obj_body.pos = np.array([delta_x, delta_y, delta_z])
                    transformed_verts[:, 0] += delta_x
                    transformed_verts[:, 1] += delta_y
                    transformed_verts[:, 2] += delta_z

                    select_pts[:, 0] += delta_x
                    select_pts[:, 1] += delta_y
                    select_pts[:, 2] += delta_z

                    max_xs.append(np.max(transformed_verts[:, 0]))
                

                    if env_config.is_debug:
                        for i in range(min(51, len(select_pts))):
                            m.body("body_vis_" + str(i)).pos = select_pts[i]

                    d.ctrl = [0, 0] # set the initial control input

                if env_config.is_debug:
                    viewers = [mujoco.viewer.launch_passive(m, d) for m, d in zip(models, datas)]
                    for v in viewers:
                        v.cam.lookat[0] = 0
                        v.cam.lookat[1] = 0
                        v.cam.lookat[2] = 0.05
                        v.cam.distance = 0.1
                else:
                    viewers = [None for _ in range(env_config.num_envs)]

                sensor_datas = [RawSensorData() for _ in range(env_config.num_envs)]

                finished_envs = [False for _ in range(env_config.num_envs)]

                cur_step = 0
                random_dest = np.random.uniform(0.5, 1)
                for s in sensor_datas:
                    s.data["whisker_vel"].append(random_dest)
                
                for s in sensor_datas:
                    s.torque_too_high = False

                all_contact_forces = []
                while True:
                    for i, (m, d, viewer, sensor_data) in enumerate(zip(models, datas, viewers, sensor_datas)):
                        if finished_envs[i]:
                            continue
                        if skipping_index[i]:
                            sensor_data.skipped = True
                            finished_envs[i] = True
                            continue
                        mujoco.mj_step(m, d)
                        transformed_force, transformed_torque = get_whisker_base_force_base_frame(m, d)
                        if cur_step>=2 and (np.max(transformed_torque * 1e7) > env_config.max_torque or np.min(transformed_torque * 1e7) < env_config.min_torque):
                            sensor_data.torque_too_high = True
                            finished_envs[i] = True
                            invalid_traj += 1
                            continue
                        elif cur_step < 2:
                            transformed_torque[-1] = 0

                        # move out and then to the origin
                        # randomness in velocity (0.003, 0.0075)
                        d.ctrl[0] = random_dest
                        if cur_step % 500 == 0:
                            delta_vel = random.uniform(-0.002, 0.002)
                        d.ctrl[1] += delta_vel

                        contact_force = np.zeros(3)
                        if len(d.contact) > 0:
                            contact_names = [
                                (m.geom(c.geom1).name, c.pos) for c in d.contact]
                            sorted(
                                contact_names, key=lambda x: int(x[0].split("G")[1]))

                            whiskerG0_pos = d.geom_xpos[m.geom("whiskerG0").id]
                            contact_pos = contact_names[0][1] - whiskerG0_pos
                            contact_force = get_contact_force(m, d)
                        else:
                            contact_pos = np.array([0,0,0])
                        
                        all_contact_forces.append(contact_force)

                        whisker_shapes = np.zeros((39, 3))
                        whiskerG0_pos = d.geom_xpos[m.geom("whiskerG0").id]
                        for k in range(39):
                            geom_name = "whiskerG" + str(k)
                            whisker_geom = m.geom(geom_name)
                            whisker_shapes[k] = d.geom_xpos[whisker_geom.id] - whiskerG0_pos
                        
                        sensor_data.append(
                            deepcopy(transformed_force), deepcopy(transformed_torque), 
                            deepcopy(contact_pos), deepcopy(d.geom_xpos[m.geom("whiskerG0").id]), deepcopy(whisker_shapes.flatten()))
                        last_whisker_positions = d.geom_xpos[m.geom("whiskerG38").id]
                        if last_whisker_positions[0] > max_xs[i]:
                            print(f"Finished simulation env {i}")
                            finished_envs[i] = True
                            
                        # Update the visualization
                        if env_config.is_debug:
                            viewer.sync()

                    if all(finished_envs):
                        break

                    cur_step += 1


                for i, sensor_data in enumerate(sensor_datas):
                    # plot contact forces
                    # plt.plot(all_contact_forces)
                    # sensor_data.plot()
                    
                    # continue
                    if not env_config.is_debug:
                        sensor_data.export_to_csv(
                            env_config.save_force, env_config.save_torque, 
                            env_config.save_path, object_name, traj_idx=traj_idx * env_config.num_envs + i,
                            sample_pts=env_config.sample_pts,
                            keep_intro_outro_0=env_config.keep_intro_outro_0)

                print(f"Finished processing {object_name} {traj_idx}")
                if env_config.is_debug:
                    for viewer in viewers:
                        viewer.close()
            except Exception as e:
                print(f"Error in processing {object_name} {traj_idx}")
                print(e)
                continue
    print(f"Invalid trajectories: {invalid_traj}")

if __name__ == "__main__":
    main()

