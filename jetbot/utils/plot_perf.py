import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os


def plot_exec_time(execution_time, model_name, model_str):
    os.environ['DISPLAY'] = ':10.0'
    execute_time = np.array(execution_time)
    mean_execute_time = np.mean(execute_time)
    max_execute_time = np.amax(execute_time)
    min_execute_time = np.amin(execute_time)

    print(
        "The execution time statistics of %s  ----- \n     Mean execution time of : %.4f sec.\n     Max execution time : %.4f sec.\n     Min execution time of : %.4f sec. " \
        % (model_name, mean_execute_time, max_execute_time, min_execute_time))

    fig = plt.figure()
    # fig, ax = plt.subplots()
    ax = fig.add_subplot()
    ax.hist(execute_time, bins=(0.005 * np.array(list(range(101)))).tolist())
    ax.set_xlabel('processing time, sec.')
    ax.set_ylabel('No. of processes')
    ax.set_title('Histogram of processing time of  ' + model_name  + "\n"+ model_str)
    props = dict(boxstyle='round', facecolor='wheat')
    text_str = " mean execution time : %.4f sec. \n max execution time : %.4f sec. \n min execution time : %.4f sec. " % (mean_execute_time, max_execute_time, min_execute_time)
    ax.text(0.6, 0.85, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    plt.show(block=False)
    