import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot Data")
    parser.add_argument("--output_file",
                        type=str,
                        default="result.png")
    parser.add_argument("--title",
                        type=str,
                        default="Learning Curve")
    parser.add_argument("--xlabel",
                        type=str,
                        default="Iteration")
    parser.add_argument("--ylabel",
                        type=str,
                        default="Return")
    parser.add_argument("--event_files",
                        type=str,
                        nargs="+")
    parser.add_argument("--names",
                        type=str,
                        nargs="+")
    parser.add_argument("--tag",
                        type=str,
                        nargs="+")
    f = parser.parse_args()

    plt.clf()
    figure = plt.figure(figsize=(10, 5))
    ax = figure.add_subplot(111)

    for tag in f.tag:

        data = {x: {'v': [], 's': []} for x in f.names}

        for i, path_to_file in enumerate(f.event_files):
            data[f.names[i]]['v'].append([])
            data[f.names[i]]['s'].append([])

            for e in tf.compat.v1.train.summary_iterator(path_to_file):
                for v in e.summary.value:
                    if tag == v.tag:
                        x = tf.make_ndarray(v.tensor)
                        data[f.names[i]]['v'][-1].append(x)
                        data[f.names[i]]['s'][-1].append(e.step)

        for name, d in data.items():

            values, steps = d['v'], d['s']

            # filter the results to be the same length
            max_length = max([len(x) for x in values])
            values = [np.pad(z, [[0, max_length - len(z)]], mode='edge') for z in values]
            steps = [z for z in steps if len(z) == max_length]

            lower = np.min(values, axis=0)
            mean = np.mean(values, axis=0)
            upper = np.max(values, axis=0)

            rgb = np.random.uniform(0.0, 0.8, size=3)

            ax.plot(
                steps[0],
                mean,
                "-",
                label=name + " " + tag,
                color=np.append(rgb, 1.0))
            ax.fill_between(
                steps[0],
                lower,
                upper,
                color=np.append(rgb, 0.2))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel(f.xlabel)
    ax.set_ylabel(f.ylabel)

    ax.set_title(f.title)
    ax.legend()
    plt.grid(True)
    plt.savefig(f.output_file)
