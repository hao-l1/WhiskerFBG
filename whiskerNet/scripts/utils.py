import numpy as np
import mujoco

def get_whisker_base_force_base_frame(m, d):
    whisker_base_force = d.sensor("force_sensor").data
    # Get the index of the site "whiskerS_first"
    sensor_index = m.body("whiskerB_first").id
    xmat_sensor = d.xmat[sensor_index].reshape(3, 3) # ^0T_1

    # Get the index of the body "dummy body"
    body_index = m.body("dummy body").id
    xmat_dummy = d.xmat[body_index].reshape(3, 3)

    # Calculate the transformation matrix from the site frame to the body frame
    transformation_matrix = xmat_dummy @ np.linalg.inv(xmat_sensor)
    # Transform the whisker_base_force to the body frame
    transformed_force = transformation_matrix @ whisker_base_force
    whisker_base_torque = d.sensor("torque_sensor").data
    # Transform the whisker_base_torque to the body frame
    transformed_torque = transformation_matrix @ whisker_base_torque

    return transformed_force, transformed_torque

def get_contact_force(m, d):
    forcetorque = np.zeros(6)
    resultant_force = np.zeros(3)
    for j, c in enumerate(d.contact):
        mujoco.mj_contactForce(m, d, j, forcetorque)
        matrix = np.array(c.frame).reshape(3,3).T
        transformed_force = matrix @ forcetorque[0:3]
        resultant_force += transformed_force[0:3]
    return resultant_force

def generate_path(num_paths=5, num_steps=15, r_array= [0.055, 0.05, 0.053, 0.054, 0.054, 0.055, 0.055, ], phi_array=[80, 85, 90, 95, 100] ,theta_array = [165, 135, 140, 145, 150, 155, 160, ], method="POLAR_COORDS"):
    _path = []
    if method == "POLAR_COORDS":
        f = lambda r, theta, phi: (
            r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi))

        for r, theta in zip(r_array, theta_array):
            for phi in phi_array:
                goal_pos = f(r=r, theta=np.deg2rad(theta), phi=np.deg2rad(phi))
                _path.append(np.linspace([0,0,0], goal_pos, num_steps+1))
    return np.array(_path)