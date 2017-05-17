""" This file defines an agent for the Box2D simulator. """
from copy import deepcopy
import numpy as np
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_VREP
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample

import vrep

class AgentVREP(Agent):
    """
    All communication between the algorithms and Box2D is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = deepcopy(AGENT_VREP)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._setup_conditions()

        self.jointHandles = np.array([-1, -1])
        self.tipHandle = -1

        self._connect_to_vrep()


    def _connect_to_vrep(self):

        # Initialise Communication Line #
        portNb = 19997
        vrep.simxFinish(-1)
        self.clientID = vrep.simxStart('127.0.0.1', portNb, True, True, 2000, 5)

        # Connect to vrep
        if self.clientID != -1:
            print('connection made')
            _, self.jointHandles[0] = vrep.simxGetObjectHandle(self.clientID, 'Shoulder', vrep.simx_opmode_blocking)
            _, self.jointHandles[1] = vrep.simxGetObjectHandle(self.clientID, 'Elbow', vrep.simx_opmode_blocking)
            _, self.tipHandle = vrep.simxGetObjectHandle(self.clientID, 'tip', vrep.simx_opmode_blocking)
            vrep.simxSynchronous(self.clientID, True)  # Enable the synchronous mode (Blocking function call)
        else:
            print('no connection')

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)
        self.x0 = self._hyperparams["x0"]

    def restart_simulation(self):
        self.stop_simulation()
        self.start_simulation()
        vrep.simxSetJointPosition(self.clientID, self.jointHandles[0], 0.75 * np.pi, vrep.simx_opmode_blocking)
        vrep.simxSetJointPosition(self.clientID, self.jointHandles[1], 0.5 * np.pi, vrep.simx_opmode_blocking)


    def stop_simulation(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)  # stop simulation
        _, sim_state = vrep.simxGetInMessageInfo(self.clientID,
                                                 vrep.simx_headeroffset_server_state)  # retrieve current sim play-pause-stop state
        while sim_state != 0:
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)  # stop simulation
            _, sim_state = vrep.simxGetInMessageInfo(self.clientID,
                                                     vrep.simx_headeroffset_server_state)  # retrieve current sim play-pause-stop state

    def start_simulation(self):
        vrep.simxSynchronous(self.clientID, True)  # Enable the synchronous mode (Blocking function call)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)  # start simulation
        _, sim_state = vrep.simxGetInMessageInfo(self.clientID,
                                                 vrep.simx_headeroffset_server_state)  # retrieve current sim play-pause-stop state
        while sim_state != 1:
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)  # stop simulation
            _, sim_state = vrep.simxGetInMessageInfo(self.clientID,
                                                     vrep.simx_headeroffset_server_state)  # retrieve current sim play-pause-stop state

    def retrieve_state(self):

        pos = np.zeros(2)
        vel = np.zeros(2)
        tip = np.zeros(3)

        _, pos[0] = vrep.simxGetJointPosition(self.clientID, self.jointHandles[0], vrep.simx_opmode_blocking)
        _, pos[1] = vrep.simxGetJointPosition(self.clientID, self.jointHandles[1], vrep.simx_opmode_blocking)
        _, vel[0] = vrep.simxGetObjectFloatParameter(self.clientID, self.jointHandles[0], 2012, vrep.simx_opmode_blocking)
        _, vel[1] = vrep.simxGetObjectFloatParameter(self.clientID, self.jointHandles[1], 2012, vrep.simx_opmode_blocking)
        _, tip_list = vrep.simxGetObjectPosition(self.clientID, self.tipHandle, -1, vrep.simx_opmode_blocking)
        tip = np.asarray(tip_list)

        return {1: pos, 2: vel, 3: tip}

    def step_simulation(self, torque_commands):

        vrep.simxSetJointTargetVelocity(self.clientID, self.jointHandles[0], 90000 if torque_commands[0] > 0 else -9000,
                                        vrep.simx_opmode_oneshot)
        vrep.simxSetJointForce(self.clientID, self.jointHandles[0], torque_commands[0] if torque_commands[0] > 0 else -torque_commands[0], vrep.simx_opmode_oneshot)

        vrep.simxSetJointTargetVelocity(self.clientID, self.jointHandles[1], 90000 if torque_commands[1] > 0 else -9000,
                                        vrep.simx_opmode_oneshot)
        vrep.simxSetJointForce(self.clientID, self.jointHandles[1], torque_commands[1] if torque_commands[1] > 0 else -torque_commands[1], vrep.simx_opmode_oneshot)

        vrep.simxSynchronousTrigger(self.clientID)  # Trigger next simulation step (Blocking function call)
        vrep.simxGetPingTime(self.clientID)  # Ensure simulation step has completed (Blocking function call)

    def sample(self, policy, condition, verbose=False, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.

        Args:
            policy: Policy to be used in the trial.
            condition (int): Which condition setup to run.
            verbose (boolean): Whether or not to plot the trial (not used here).
            save (boolean): Whether or not to store the trial into the samples.
            noisy (boolean): Whether or not to use noise during sampling.
        """

        print('taking sample')

        self.restart_simulation()

        vrep_X = self.retrieve_state() # get simulation state
        new_sample = self._init_sample(vrep_X) # initialise sample with this world state at initial time step
        U = np.zeros([self.T, self.dU]) # initialise episode action vector dims with episode time span and action space dim

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams) # Generate a T x dU gaussian-distributed noise vector
        else:
            noise = np.zeros((self.T, self.dU)) # vector of zeros

        for t in range(self.T): # iterate over episode time-steps
            X_t = new_sample.get_X(t=t) # get state vector of joint angles, joint velocities, and 3D end-effector point concatended for current time-step
            obs_t = new_sample.get_obs(t=t) # get NULL observation for simple trajectory optimisation
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :]) # return action for current state, and fill in entry of U vector for time step t
            if (t+1) < self.T: # provided we are not on the final iteration of the for loop
                for _ in range(self._hyperparams['substeps']): # iterate over sub_steps, (i.e. how much frame skipping is there for repeating actions)
                    self.step_simulation(U[t, :])
                vrep_X = self.retrieve_state()  # get simulation state
                self._set_sample(new_sample, vrep_X, t) # add this information to new_sample object
        self.stop_simulation()
        new_sample.set(ACTION, U) # having run to end of trajectory, set saved action vector U as ACTION data in new_sample _data, to provide full sample information in new_sample object
        if save: # if want to save sample to agent object
            self._samples[condition].append(new_sample) # append samples to agent object _samples variable

        return new_sample # 0: actions 1: joint angles 2: joint velocities 3: end-effector points (3D)

    def _init_sample(self, b2d_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, b2d_X, -1)
        return sample

    def _set_sample(self, sample, b2d_X, t):
        for sensor in b2d_X.keys():
            sample.set(sensor, np.array(b2d_X[sensor]), t=t+1)

