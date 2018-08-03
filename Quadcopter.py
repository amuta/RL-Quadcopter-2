import os
from DDPG import DDPG
import numpy as np
import matplotlib.pyplot as plt
from task import Task
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import set_random_seed

np.random.seed(1)
set_random_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # forces tensorflow to use CPU


class Quadcopter():

    def __init__(self):
        self.env = self.new_env()
        self.agent = DDPG(self.env)
        self.train_hist = []
        self.test_hist = []

    def new_env(self):
        init_pose = np.array([0., 0., 3., 0., 0., 0.])
        target_pos = np.array([0., 0., 50.])
        return Task(init_pose=init_pose, target_pos=target_pos)

    def preprocess_state(self, state):
        return state

    def plot_Q(self, xy_step=1, z_step=2, action_step=150):
        """
        Plots 4 heatmaps that shows the behavior of the
        local critic and target when dealing with the state
        and actions space
        """
        xy_plot_range = np.arange(-10, 10 + xy_step, xy_step)
        z_plot_range = np.arange(0, 100 + z_step, z_step)
        action_range = np.arange(0, 900 + action_step, action_step)
        shape = (z_plot_range.shape[0], xy_plot_range.shape[0])
        xz_matrix_mQ = np.ones(shape)
        xz_matrix_sQ = np.ones(shape)
        yz_matrix_mQ = np.ones(shape)
        yz_matrix_sQ = np.ones(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                xy = xy_plot_range[j]
                z = z_plot_range[i]
                x_state = np.array([xy, 0, z, 0, 0, 0, 0, 0, 0]).reshape(-1, 9)
                y_state = np.array([0, xy, z, 0, 0, 0, 0, 0, 0]).reshape(-1, 9)

                xz_Q_list = []
                yz_Q_list = []
                for a in action_range:
                    action = np.array([0, 0, 0, 0]).reshape(-1, 4)  # np.array([a] * 4).reshape(-1, 4)
                    xz_Q_list.append(self.agent.critic_local.model.predict(
                                     [x_state, action]))
                    yz_Q_list.append(self.agent.critic_local.model.predict(
                                     [y_state, action]))

                xz_matrix_mQ[i][j] = np.max(xz_Q_list)
                xz_matrix_sQ[i][j] = np.sum(xz_Q_list)
                yz_matrix_mQ[i][j] = np.max(yz_Q_list)
                yz_matrix_sQ[i][j] = np.sum(yz_Q_list)
                # matrix_A[i][j] = agent.actor_local.model.predict(state)
        extent = [10, -10, 0, 100]

        fig, ax = plt.subplots(2, 2, sharex=True)
        ax[0, 0].set_title('Q value max')
        ax[0, 0].set_ylabel('Z-pos')
        ax[0, 0].set_xlabel('X-pos')
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[0, 0].imshow(xz_matrix_mQ, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        ax[0, 1].set_title('Q value sum')
        ax[0, 1].set_ylabel('Z-pos')
        ax[0, 1].set_xlabel('X-pos')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[0, 1].imshow(xz_matrix_sQ, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        # ax[1, 0].set_title('Q value max')
        ax[1, 0].set_ylabel('Z-pos')
        ax[1, 0].set_xlabel('Y-pos')
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[1, 0].imshow(yz_matrix_mQ, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        # ax[1, 1].set_title('Q value sum')
        ax[1, 1].set_ylabel('Z-pos')
        ax[1, 1].set_xlabel('Y-pos')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[1, 1].imshow(yz_matrix_sQ, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        plt.subplots_adjust(top=0.92, right=0.95, hspace=0.25, wspace=0.4, bottom=0.2)

        plt.show()

    def run_epoch(self, max_steps, render=False, training=True):
        self.env = self.new_env()
        state = self.preprocess_state(self.env.reset())
        self.agent.reset_episode(state)
        actions_list = []
        sim_info = []
        total_reward = 0
        steps = 0
        while steps < max_steps:
            steps += 1
            noisy_action, pure_action = self.agent.act(state)

            # use action with OUNoise if training
            action = noisy_action if training else pure_action

            # step into the environment and update values
            next_state, reward, done = self.env.step(action)
            sim_info.append(self.env.sim.pose.tolist() + self.env.sim.v.tolist())
            next_state = self.preprocess_state(next_state)
            state = next_state
            total_reward += reward
            actions_list.append(action)

            # only train agent if in training
            if training:
                self.agent.step(action, reward, next_state, done)

            if done:
                break

        action_mean = np.mean(actions_list, axis=0)
        action_std = np.std(actions_list, axis=0)

        return total_reward, done, action_mean, action_std, steps, sim_info

    def run_model(self, max_epochs=100, n_solved=1, r_solved=250,
                  max_steps=1000, verbose=1):
        """
        Train the learner

        Params
            ======
                max_epochs (int): Maximum number of training episodes
                max_steps (int): Maximum steps in each episode
                r_solved (int): Minimum reward value to consider episode solved
                n_solved (int): Targed number of solved episodes before break
                plot_Q (bool): If true will plot state action values heatmaps
                verbose (int): How much information each epoch will print,
                  possible values are 1,0 and-1 in order of verbosity
        """
        for epoch in range(1, max_epochs + 1):
            train_reward, train_done, train_action_mean, train_action_std, \
                train_steps, train_infos = self.run_epoch(max_steps=max_steps)
            test_reward, test_done, test_action_mean, test_action_std, \
                test_steps, test_infos = self.run_epoch(max_steps=max_steps,
                                                        training=False)

            self.train_hist.append([train_reward, train_steps])
            self.test_hist.append([test_reward, test_steps])

            if epoch > n_solved:
                train_running = np.mean([r for r, s in self.train_hist][-n_solved:])
                test_running = np.mean([r for r, s in self.test_hist][-n_solved:])
            else:
                train_running = np.mean([r for r, s in self.train_hist])
                test_running = np.mean([r for r, s in self.test_hist])

            print_vals = {
                'epoch': epoch,
                'train_reward': train_reward,
                'train_steps': train_steps,
                'train_running': train_running,
                'train_action_mean': train_action_mean.tolist(),
                'train_action_std': train_action_std.tolist(),
                'train_fpose/v': train_infos[-1:][0],
                'test_reward': test_reward,
                'test_steps': test_steps,
                'test_running': test_running,
                'test_action_mean': test_action_mean.tolist(),
                'test_action_std': test_action_std.tolist(),
                'test_fpose/v': test_infos[-1:][0]
            }

            self.print_epoch(print_vals, verbose)

            if test_running > r_solved:
                print('\nSolved in epoch {:4d}'.format(epoch))
                break

    def print_epoch(self, vals, verbose):
        print('===========================')
        if verbose == 1:
            for key, values in vals.items():
                if isinstance(values, (list,)):
                    vals_str = ''.join(s for s in ['{:6.2f} '.format(s) for s in values])
                    print(key, ':', vals_str)
                else:
                    print(key, ':', values)
        elif verbose == 0:
            print('Epoch: {:4d} Train Reward: {:6.2f} Test Reward: {:6.2f}'
                  ''.format(vals['epoch'], vals['train_reward'], vals['test_reward']),
                  end='\r')


if __name__ == '__main__':
    print('Running learner directly')
    Learner = Quadcopter()
    Learner.run_model(max_epochs=500, n_solved=1, r_solved=250, verbose=1)
    Learner.plot_Q()
