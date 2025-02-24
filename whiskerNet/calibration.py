import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import time
from scripts.utils import generate_path, get_contact_force, get_whisker_base_force_base_frame
from copy import deepcopy
from collections import deque
from tqdm import tqdm
import pandas as pd
import hydra


@hydra.main(version_base=None, config_path="./config", config_name="calibration")
def main(cfg):
    env_config = cfg.environment

    # Load your MuJoCo XML model and create a simulation
    m = mujoco.MjModel.from_xml_path(env_config.xml_file)
    d = mujoco.MjData(m)

    # zero gravity
    m.opt.gravity = (0,0,0)
    mujoco.mj_resetData(m, d)

    d.ctrl = [0,0,0] # set the initial control input

    fixture = d.body("fixture")
    fixture_pos = fixture.xpos

    # relative pose from fixture left bottom to the tip
    m.body("fixture").pos = env_config.tip_pos
    steady_threshold = env_config.steady_threshold

    r_array = env_config.r_array
    phi_array = env_config.phi_array
    theta_array = env_config.theta_array

    paths = generate_path(num_steps=env_config.num_steps, r_array=r_array, phi_array=phi_array, theta_array=theta_array, method=env_config.method)

    step_pts = paths.reshape(-1,3)

    if env_config.is_debug:
        viewer = mujoco.viewer.launch_passive(m, d)
    results = {
        "force": [],
        "torque": [],
        "shape": [],
        "contact_pos": [],
        "step_pt": []
    }
    fixture_pos_buffer = deque(maxlen=10)
    no_contact = 0
    time.sleep(5)
    for cur_pts in tqdm(range(len(step_pts))):
        while True:
            mujoco.mj_step(m, d)
            # move out and then to the origin
            if cur_pts != 0 and all(step_pts[cur_pts] == 0):
                d.ctrl[0] = 0 
                d.ctrl[1] = - step_pts[cur_pts-1][1] - 0.06
                d.ctrl[2] = 0
                for i in range(10):
                    mujoco.mj_step(m, d)
                    if env_config.is_debug:
                        viewer.sync()

            # boundary control x: -0.034, y: -0.015, z: 0.02
            d.ctrl[0] = step_pts[cur_pts][0]
            d.ctrl[1] = -step_pts[cur_pts][1]
            d.ctrl[2] = step_pts[cur_pts][2]

            fixture_pos_buffer.append(deepcopy(fixture.xpos))
            is_steady = len(fixture_pos_buffer) == fixture_pos_buffer.maxlen and \
                np.linalg.norm(np.array(fixture_pos_buffer[0]) - np.array(fixture_pos_buffer[-1])) < steady_threshold
            
            if is_steady:
                for i in range(4000):
                    mujoco.mj_step(m, d)
                    if env_config.is_debug:
                        viewer.sync()
                transformed_force, transformed_torque = get_whisker_base_force_base_frame(m, d)
                if cur_pts%(env_config.num_steps+1) == 0:
                    break
                # average contact position
                contact_pos =  np.mean(np.array([c.pos for c in d.contact]), axis=0) if \
                    len(d.contact) > 0 else np.array([0,0,0]) 
                if len(d.contact) == 0:
                    no_contact += 1
                    print("No contact")
                
                whisker_positions = np.zeros((39, 3))
                for i in range(39):
                    geom_name = "whiskerG" + str(i)
                    geom = m.geom(geom_name)
                    whisker_positions[i] = d.geom_xpos[geom.id]
                results["force"].append(deepcopy(transformed_force))       
                results["torque"].append(deepcopy(transformed_torque))
                results["contact_pos"].append(deepcopy(contact_pos))
                results["step_pt"].append(deepcopy(fixture.xpos - env_config.tip_pos))
                break

            # Update the visualization
            if env_config.is_debug:
                viewer.sync()

    print(f"No contact: {no_contact}")

    if env_config.save_force:
        # Save the results
        results_all = np.hstack([results["contact_pos"], results["step_pt"], np.array(results["force"]) ])
        save_path = env_config.save_path.replace("timestamp", time.strftime("%Y%m%d-%H%M%S"))
        save_path = save_path[:-5]+ "_force.csv"
        # add title px, py, pz, sx, sy, sz
        
        df = pd.DataFrame(results_all, columns=['px', 'py', 'pz','stpx','stpy','stpz','sx', 'sy', 'sz'])
        df.to_csv(save_path, index=True)
        print(f"Saved to {save_path}")
    if env_config.save_torque:
        # Save the results
        results_all = np.hstack([results["contact_pos"], results["step_pt"], np.array(results["torque"]) * 1e7])
        save_path = env_config.save_path.replace("timestamp", time.strftime("%Y%m%d-%H%M%S"))
        save_path = save_path[:-5]+ "_torque.csv"
        # add title px, py, pz, sx, sy, sz
        
        df = pd.DataFrame(results_all, columns=['px', 'py', 'pz','stpx','stpy','stpz', 'sx', 'sy', 'sz'])
        df.to_csv(save_path, index=True)
        print(f"Saved to {save_path}")
    
    exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        exit(0)

