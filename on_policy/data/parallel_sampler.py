from on_policy.data.sampler import Sampler, identity
from on_policy import init_process
import time
import multiprocessing
import tree
import os
import numpy as np


map = tree.map_structure_up_to
concat = np.concatenate


def launch_worker(set_weights_queue,
                  sample_queue,
                  out_queue,
                  *args,
                  **kwargs):
    """Creates a python process that loops forever and responds to
    inputs via multiprocessing queues

    Arguments:

    set_weights_queue: Queue
        a queue that provides updates weights to use to collect
        samples using new policies
    sample_queue: Queue
        a queue that accepts the number of samples to be
        collected using the current agent
    out_queue: Queue
        a queue that will contain samples returned bvy the worker
        processes when collecting data"""

    init_process()
    worker = Sampler(*args, **kwargs)
    while True:
        if not set_weights_queue.empty():
            worker.set_weights(set_weights_queue.get())
        elif not sample_queue.empty():
            a, b, c = sample_queue.get()
            out_queue.put(worker.sample(a, deterministic=b, render=c))
        else:
            time.sleep(0.05)


class ParallelSampler(object):

    def __init__(self,
                 env,
                 agent,
                 num_workers=10,
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
        num_workers: int
            the number of parallel sampling processes to use when collecting
            roll outs to train the agent
        max_horizon: int
            the maximum number of roll out steps before an episode terminates
            can be set to infinity
        act_selector: Callable
            a function that selects into a possibly nested structure of
            actions output from the policy
        obs_spec: Any
            a nested structure with the same topology as the observations
            from the environment
        act_spec: Any
            a nested structure with the same topology as the actions
            from the policy"""

        self.num_workers = num_workers
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.kwargs = dict(max_horizon=max_horizon,
                           act_selector=act_selector,
                           obs_spec=obs_spec,
                           act_spec=act_spec)

        # create queues to pass messages between sampling processes
        self.set_weights_queues = [
            multiprocessing.Queue() for i in range(num_workers)]
        self.sample_queues = [
            multiprocessing.Queue() for i in range(num_workers)]
        self.out_queues = [
            multiprocessing.Queue() for i in range(num_workers)]

        # create parallel worker sampling processes
        self.p = [multiprocessing.Process(
            target=launch_worker,
            kwargs=self.kwargs,
            args=(self.set_weights_queues[i],
                  self.sample_queues[i],
                  self.out_queues[i],
                  env,
                  agent)) for i in range(num_workers)]

    def __enter__(self):
        """Start each of the worker processes and begin waiting for
        instructions for collecting data"""

        [process.start() for process in self.p]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Start each of the worker processes and begin waiting for
        instructions for collecting data"""

        [process.terminate() for process in self.p]

    def set_weights(self,
                    weights):
        """Sets the weights of the policy to the provided weights,
        which are typically from another process

        Arguments:

        weights: list
            a list of variables that can be obtained by calling
            model.get_weights on a copy of self.policy"""

        for q in self.set_weights_queues:
            q.put(weights)

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
        render: bool or str
            determines if the environment renders the state and
            behavior of the agent during a roll out; if a string
            then dump a video instead of displaying

        Returns:

        observations: np.ndarray
            a tensor containing observations experienced by the agent
            that is shaped like [batch_dim, obs_dim]
        actions: np.ndarray
            a tensor containing actions taken by the agent
            that is shaped like [batch_dim, act_dim]
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

        # calculate how many samples to collect per worker
        for i, q in enumerate(self.sample_queues):
            per_worker = min_num_steps // self.num_workers
            per_worker += 1 if i < min_num_steps % self.num_workers else 0
            q.put([per_worker, deterministic, render if i == 0 else None])

        # create a buffer to store incoming data
        out_o = tree.map_structure(lambda _: [], self.obs_spec)
        out_a = tree.map_structure(lambda _: [], self.act_spec)
        out_ret = tree.map_structure(lambda _: [], self.act_spec)
        out_adv = tree.map_structure(lambda _: [], self.act_spec)

        out_rew = []
        out_lengths = []

        # keep track of which of the workers has finished sp far
        open_set = set(range(self.num_workers))

        # collect data when a worker pushes it into the queue
        while len(open_set) > 0:
            time.sleep(0.05)
            for i in set(open_set):
                if not self.out_queues[i].empty():
                    o, a, ret, adv, rew, lengths = self.out_queues[i].get()
                    open_set.remove(i)

                    # add the worker result to an output buffer
                    map(self.obs_spec,
                        lambda x, y: x.append(y), out_o, o)
                    map(self.act_spec,
                        lambda x, y: x.append(y), out_a, a)
                    map(self.act_spec,
                        lambda x, y: x.append(y), out_ret, ret)
                    map(self.act_spec,
                        lambda x, y: x.append(y), out_adv, adv)

                    out_rew.append(rew)
                    out_lengths.append(lengths)

        # concatenate samples into a contiguous batch
        out_o = map(self.obs_spec,
                    lambda x: concat(x, axis=0), out_o)
        out_a = map(self.act_spec,
                    lambda x: concat(x, axis=0), out_a)
        out_ret = map(self.act_spec,
                      lambda x: concat(x, axis=0), out_ret)
        out_adv = map(self.act_spec,
                      lambda x: concat(x, axis=0), out_adv)

        out_rew = concat(out_rew, axis=0)
        out_lengths = concat(out_lengths, axis=0)

        return (out_o,
                out_a,
                out_ret,
                out_adv,
                out_rew,
                out_lengths)
