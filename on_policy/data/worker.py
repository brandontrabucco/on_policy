import numpy as np
import tree


map = tree.map_structure_up_to
concat = np.concatenate


def identity(x):
    """A default function that can be pickled"""

    return x


class Worker(object):

    def __init__(self,
                 env,
                 agent,
                 max_horizon=1000,
                 act_selector=identity,
                 obs_spec=None,
                 act_spec=None):
        """Creates a sampler for collecting roll outs for training an
        on policy rl algorithm

        Arguments:

        env: gym.Env:
            an instance of an OpenAI Gym environment for collecting samples
            from, the environment must be pickle-able
        agent: PicklingModel
            a keras model wrapped with a helper class that enables the model
            to be pickled and sent between threads
        max_horizon: int
            the maximum number of roll out steps before an episode terminates
            can be set to infinity
        obs_selector: Callable
            a function that selects into a possibly nested observation
            from the environment
        obs_spec: Any
            a nested structure with the same topology as the observations
            from the environment
        act_spec: Any
            a nested structure with the same topology as the actions
            from the policy"""

        self.env = env
        self.agent = agent
        self.max_horizon = max_horizon

        self.act_selector = act_selector
        self.obs_spec = obs_spec
        self.act_spec = act_spec

    def set_weights(self,
                    weights):
        """Sets the weights of the policy to the provided weights,
        which are typically from another process

        Arguments:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on a copy of self.policy"""

        self.agent.set_weights(weights)

    def sample(self,
               min_num_steps,
               deterministic=False,
               render=False):
        """Samples roll outs from the environment and return the
        roll outs to train an on policy rl algorithm

        Arguments:

        min_num_steps: int
            the minimum number of steps to collect from the
            environment in this set of samples
        deterministic: bool
            collects samples using the exploration policy if true
            and using the evaluation policy otherwise
        render: bool
            determines if the environment renders the state and
            behavior of the agent during a roll out

        Returns:

        observations: np.ndarray
            a tensor containing observations experienced by the agent
            that is shaped like [batch_dim, obs_dim]
        actions: np.ndarray
            a tensor containing actions taken by the agent
            that is shaped like [batch_dim, act_dim]
        log_probs: np.ndarray
            a tensor containing the log probabilities of the actions
            taken by the agent during a roll out
            that is shaped like [batch_dim]
        returns: np.ndarray
            a tensor containing returns experienced by the agent
            that is shaped like [batch_dim]
        advantages: np.ndarray
            a tensor containing advantages estimated by the agent
            that is shaped like [batch_dim]
        rewards: np.ndarray
            a tensor containing rewards experienced by the agent
            that is shaped like [batch_dim]
        lengths: np.ndarray
            a tensor containing the episode lengths experienced
            by the agent for splitting later on
            shaped like [num_episodes]"""

        end_of_episode = True
        time_step = 0

        # the actions and observations can be structures
        observations = []
        actions = []
        rewards = []

        # collect precisely min_num_steps numbers of transitions
        for step in range(min_num_steps):

            # called at the beginning of each episode
            if end_of_episode:
                o = self.env.reset()
                time_step = 0

                observations.append(map(self.obs_spec, lambda x: [x], o))
                actions.append(tree.map_structure(lambda _: [], self.act_spec))
                rewards.append([])

            # select the right observation for the policy
            o = map(self.obs_spec,
                    lambda x: x[-1][np.newaxis, ...],
                    observations[-1])

            # choose to apply the exploration or evaluation policy
            if deterministic:
                a = self.agent.expected_value(time_step, o)
            else:
                a = self.agent.sample(time_step, o)

            # remove the batch dimension
            a = map(self.act_spec, lambda x: x[0], a)

            # apply atomic actions in the environment
            o, r, end_of_episode, _ = self.env.step(self.act_selector(a))
            time_step += 1

            # render the environment
            if render:
                self.env.render()

            # store the transition sampled from the environment
            observations[-1].append(o)
            rewards[-1].append(r)
            actions[-1].append(a)

            # terminate if the episode has been running too long
            if end_of_episode:
                rewards[-1].append(0.0)

            # terminate if the episode has been running too long
            elif time_step >= self.max_horizon or step >= min_num_steps - 1:
                end_of_episode = True

                # select the right observation for the policy
                last_o = map(self.obs_spec,
                             lambda x: x[-1][np.newaxis, ...],
                             observations[-1])

                # bootstrap the reward-to-go to account for time steps
                # beyond the arbitrary episode horizon
                # https://github.com/openai/spinningup/blob/master
                # /spinup/algos/tf1/ppo/ppo.py#L41
                rewards[-1].append(self.agent.get_values(last_o)[0])

        out_o = tree.map_structure(lambda _: [], self.obs_spec)
        out_a = tree.map_structure(lambda _: [], self.act_spec)
        out_p = tree.map_structure(lambda _: [], self.act_spec)
        out_ret = tree.map_structure(lambda _: [], self.act_spec)
        out_adv = tree.map_structure(lambda _: [], self.act_spec)
        out_rew = []
        out_lengths = np.array([len(r) - 1 for r in rewards])

        # post process the sampled data and convert into
        # a standard format for training
        for path_r, path_a, path_o in zip(rewards,
                                          actions,
                                          observations):

            # convert samples to float32
            path_r = np.array(path_r, np.float32)
            path_o = map(self.obs_spec,
                         lambda x: np.array(x, np.float32),
                         path_o)

            # create labels for the returns and generalized advantages
            path_ret = self.agent.get_returns(path_r)[:-1]
            path_adv = self.agent.get_advantages(path_r, path_o)

            # remove the final observation that is unused
            path_o = path_o[:-1]
            path_a = map(self.act_spec, lambda x: np.array(x), path_a)
            path_p = self.agent.log_prob(np.arange(path_ret.shape[0]),
                                         path_a,
                                         path_o)

            # add the processed samples to the return buffer
            map(self.obs_spec,
                lambda x, y: x.append(y), out_o, path_o)
            map(self.act_spec,
                lambda x, y: x.append(y), out_a, path_a)
            map(self.act_spec,
                lambda x, y: x.append(y), out_p, path_p)
            map(self.act_spec,
                lambda x, y: x.append(y), out_ret, path_ret)
            map(self.act_spec,
                lambda x, y: x.append(y), out_adv, path_adv)
            out_rew.append(path_r[:-1])

        # concatenate samples into a contiguous batch
        out_o = map(self.obs_spec,
                    lambda x: concat(x, axis=0), out_o)
        out_a = map(self.act_spec,
                    lambda x: concat(x, axis=0), out_a)
        out_p = map(self.act_spec,
                    lambda x: concat(x, axis=0), out_p)
        out_ret = map(self.act_spec,
                      lambda x: concat(x, axis=0), out_ret)
        out_adv = map(self.act_spec,
                      lambda x: concat(x, axis=0), out_adv)
        out_rew = concat(out_rew, axis=0)

        return (out_o,
                out_a,
                out_p,
                out_ret,
                out_adv,
                out_rew,
                out_lengths)
