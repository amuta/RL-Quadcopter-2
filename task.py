import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 1

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.done = False

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_updates(self):
        # # state xyz position
        sim_xyz = self.sim.pose[:3].tolist()
        tgt_xyz = self.target_pos.tolist()
        relative_vel = self.sim.v.tolist()
        state =  self.sim.pose.tolist() + relative_vel


        reward = np.clip(relative_vel[2]*0.5, -1, 1)

        if np.abs(sim_xyz[0] - tgt_xyz[0]) > 5 or np.abs(sim_xyz[1] - tgt_xyz[1]) > 5:
            self.done = True
            reward = -100
        elif np.abs(sim_xyz[2]) < 1:
            self.done = True
            reward = -100

        return list(state), reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        sta_all = []
        for _ in range(self.action_repeat):
            # rotor_speeds = [900,900,900,900]
            self.done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            s, r = self.get_updates()
            sta_all.append(s)
            reward += r
        next_state = np.concatenate(sta_all)
        return next_state, reward, self.done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state, _ = self.get_updates()
        state = state * self.action_repeat
        return state
