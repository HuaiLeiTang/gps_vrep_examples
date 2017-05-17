""" This file defines the main object that runs experiments. """

# COMMENTS BASED ON SIMPLE TRAJECTORY OPTIMISATION #
# NOT FULL GUIDED POLICY SEARCH ALGORITHM #

import matplotlib as mpl # import for plotting
mpl.use('Qt4Agg') # rendering type

import logging # for event logging
import imp # for importing
import os # for os functionaility
import os.path # for os pathname functions
import sys # for functionaility with interpreter
import copy # for copying
import argparse # for parsing arguments
import threading # for threading
import time # for timing
import traceback # for outputing stack traces

sys.path.append('/home/djl11/Sources/gps/python') # Add gps/python to path so that imports work.
from gps.gui.gps_training_gui import GPSTrainingGUI # Graphical User Interface for GPS training
from gps.utility.data_logger import DataLogger # for pickling data into files and unpickling data from files.
from gps.sample.sample_list import SampleList # for handling reads/writes, pickling, and notifications for sample data


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """

        # VARIABLE INITIALISATION #

        self._quit_on_end = quit_on_end
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        try:
            itr_start = self._initialize(itr_load) # perform initialization, and return starting iteration

            for itr in range(itr_start, self._hyperparams['iterations']): # iterations of GPS
                for cond in self._train_idx: # number of distinct problem instantiations for training
                    for i in range(self._hyperparams['num_samples']): # number of gps samples
                        self._take_sample(itr, cond, i) # take sample from LQR controller, and store in agent object

                traj_sample_lists = [
                    self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                    for cond in self._train_idx
                ] # store list of trajectory samples, for all training conditions on this iteration

                # Clear agent samples.
                self.agent.clear_samples()

                self._take_iteration(itr, traj_sample_lists) # take iteration, updating dynamics model, and minimising cost of LQR controller
                pol_sample_lists = self._take_policy_samples() # take samples of current LQR policy, to see how it's doing
                self._log_data(itr, traj_sample_lists, pol_sample_lists) # logs samples from the policy before and after the iteration taken, and the algorithm object, into pickled bit-stream files
        except Exception as e:
            traceback.print_exception(*sys.exc_info()) # else, catch exception
        finally: # ensures exexcution within try statement, whether exception occured or not
            self._end() # show process ended in gui, and close python if user argument selected as such

    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        ) # save the policy samples as a pickled bit-stream file

        if self.gui: # if using gui
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None: # if no iteration to load from specified
            if self.gui: # if using gui
                self.gui.set_status_text('Press \'go\' to begin.') # inform user to start the process
            return 0 # return iteration number as 0, ie start from beginning
        else:
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load # define algorithm file based on desired iteration
            self.algorithm = self.data_logger.unpickle(algorithm_file) # Read string from file and interpret as pickle data stream, reconstructing and returning the original object
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread

            if self.gui: # if using gui
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load)) # unpickle traj sample lists for current iterations
                if self.algorithm.cur[0].pol_info: # if policy info available
                    pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                        ('pol_sample_itr_%02d.pkl' % itr_load)) # unpickle policy sample lists for current iteration
                else:
                    pol_sample_lists = None
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load) # inform user to start the process
            return itr_load + 1 # return iteration number to begin working from

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self.algorithm._hyperparams['sample_on_policy'] \
                and self.algorithm.iteration_count > 0:
            pol = self.algorithm.policy_opt.policy # There is NO policy optimisation for simple traj opt algorithm
        else:
            pol = self.algorithm.cur[cond].traj_distr # initialise LQR controller based on current trajectories
            # Note: policy.act(array(7)->X, array(empty)->obs, int->t, array(2)->noise) to retrieve policy actions conditioned on state
        if self.gui: # if using gui
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                # For Handling GUI Requests 'stop', 'reset', 'go', 'fail' #
                while self.gui.mode in ('wait', 'request', 'process'): # while gui is waiting, requesting, or processing
                    if self.gui.mode in ('wait', 'process'): # if waiting or processing,
                        time.sleep(0.01)
                        continue # continue again at start of while loop
                    # 'request' mode.
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                ) # display to user
                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                ) # run trail using policy, and save trajectory into agent object (AgentBox2D or AgentMuJoCo class)

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()  # Complete request.
                    self.agent.delete_last_sample(cond) # delete last sample, and redo, on account of fail request
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials'])
            ) # run trail using policy, and save trajectory into agent object (AgentBox2D or AgentMuJoCo class)

    def _take_iteration(self, itr, sample_lists):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()
        self.algorithm.iteration(sample_lists) # take iteration, training LQR controller and updating dynamics model
        if self.gui:
            self.gui.stop_display_calculating()

    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials'] # bool as to whether to use verbose trails or not
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.algorithm.policy_opt.policy, self._test_idx[cond],
                verbose=verbose, save=False, noisy=False) # iterate through problem instantiations, accumulating policy samples from LQR policy
        return [SampleList(samples) for samples in pol_samples] # return samples, held in SampleList objects

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if self.gui: # if using gui
            self.gui.set_status_text('Logging data and updating GUI.')
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        ) # store current algorithm iteration object as bit-stream file
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        ) # store current trajectory sample list as bit-stream file
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            ) # store current policy sample list as bit-stream file

    def _end(self):
        """ Finish running and exit. """
        if self.gui: # if using gui
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
            if self._quit_on_end: # user argument for main
                # Quit automatically (for running sequential expts)
                os._exit(1) # exit python altogether

def main():

    # INPUT ARGUMENTS #

    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('-e', '--experiment', type=str, default='box2d_arm_example',
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM/MDGPS only)')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    # INPUT VARIABLES #

    exp_name = args.experiment # experiment name
    resume_training_itr = args.resume # iteration from which to resume training
    test_policy_N = args.policy # number of policy samples to take

    # FILE-PATHS #

    from gps import __file__ as gps_filepath # set 'gps_filepath' as root gps filepath
    gps_filepath = os.path.abspath(gps_filepath) # reformat as absolute path
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/' # remove 'gps' part, for root directory
    exp_dir = gps_dir + 'experiments/' + exp_name + '/' # create experiment directory
    hyperparams_file = __file__[:-7] + 'hyperparams.py' # complete path to hyperparameter file

    # LOGGING OPTION #

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


    if args.new: # if new experiment desired
        from shutil import copy # import file copy

        if os.path.exists(exp_dir): # if already exists
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir)) # exit from python
        os.makedirs(exp_dir) # else mkdir

        prev_exp_file = '.previous_experiment' # hidden file in python gps directory, IF by previous run
        prev_exp_dir = None
        try: # attempt following code
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline() # read previous experiment directory from hidden file
            copy(prev_exp_dir + 'hyperparams.py', exp_dir) # copy over hyperparameters from previous exp run
            if os.path.exists(prev_exp_dir + 'targets.npz'): # if target numpy array file exists
                copy(prev_exp_dir + 'targets.npz', exp_dir) # copy to new experiment directory
        except IOError as e: # throw program terminating exception, unless IOError
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
                # if hyperparams were not copied over, instruct user on how to get started
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir) # regardless of whether existed before, write new prev_exp hidden file

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file)) # base output message
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg) # if hyperparam file copied, also state where from
        # Finally, exit process, new experiment has been created, can now run again without '-n' argument

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file)) # if no hyperparams file, prompt to create one, and exit

    hyperparams = imp.load_source('hyperparams', hyperparams_file) # import hyperparams from hyperparam file

    if args.targetsetup: # if target setup GUI option selected (for ROS only)
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros.agent_ros import AgentROS
            from gps.gui.target_setup_gui import TargetSetupGUI

            agent = AgentROS(hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
    elif test_policy_N: # if testing current policy, how many policy samples to take at given iteration
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0) # retrieve random_seed value from hyperparams
        random.seed(seed) # initialize internal state of random num generator with fixed seed
        np.random.seed(seed) # initialize internal state of numpy random num generator with fixed seed

        data_files_dir = exp_dir + 'data_files/' # data file dir
        data_filenames = os.listdir(data_files_dir) # all data files
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)] # all algorithm iteration files
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0] # current algorithm iteration filename
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2]) # current iteration number

        gps = GPSMain(hyperparams.config) # initialise GPSMain object
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            ) # define thread target (what is called on 'start' command)
            test_policy.daemon = True # daemon threads are killed automatically on program termination
            test_policy.start() # start thread process

            plt.ioff() # turn interactive mode off
            plt.show() # start mainloop for displaying plots
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N) # else, no seperate thread needed, start process as normal
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0) # retrieve random_seed value from hyperparams
        random.seed(seed) # initialize internal state of random num generator with fixed seed
        np.random.seed(seed) # initialize internal state of numpy random num generator with fixed seed

        gps = GPSMain(hyperparams.config, args.quit) # initialise GPSMain object
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            ) # define thread target (what is called on 'start' command)
            run_gps.daemon = True # daemon threads are killed automatically on program termination
            run_gps.start() # start thread process

            plt.ioff() # turn interactive mode off
            plt.show() # start mainloop for displaying plots
        else:
            gps.run(itr_load=resume_training_itr) # else, no seperate thread needed, start process as normal


if __name__ == "__main__":
    main()