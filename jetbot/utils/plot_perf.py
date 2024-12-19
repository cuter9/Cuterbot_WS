import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

font = {'fontweight': 'normal', 'fontsize': 12}
font_title = {'fontweight': 'medium', 'fontsize': 18}


def plot_exec_time(execution_time, model_name, model_str):
    # os.environ['DISPLAY'] = ':10.0'
    execute_time = np.array(execution_time)
    mean_execute_time = np.mean(execute_time)
    max_execute_time = np.amax(execute_time)
    min_execute_time = np.amin(execute_time)

    # fps = np.array(fps)
    mean_fps = 1 / mean_execute_time
    max_fps = 1 / min_execute_time
    min_fps = 1 / max_execute_time

    print(
        "The execution time statistics of %s  ----- \n     Mean execution time of : %.4f sec.\n     Max execution time : %.4f sec.\n     Min execution time of : %.4f sec. " \
        % (model_name, float(mean_execute_time), float(max_execute_time), float(min_execute_time)))

    # plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))

    nbin = 150
    sbin = (max_execute_time * 1.2 - min_execute_time * 0.8) / nbin
    ax.hist(execute_time, bins=(np.arange(min_execute_time * 0.8, max_execute_time * 1.2, sbin)).tolist())
    # ax.hist(execute_time, bins=(0.003 * np.array(list(range(151)))).tolist())
    ax.set_xlabel('processing time, sec.', fontdict=font)
    ax.set_ylabel('No. of processes', fontdict=font)
    ax.set_title('Histogram of processing time of  ' + model_name + "\n" + model_str, fontdict=font_title)
    props = dict(boxstyle='round', facecolor='wheat')
    text_str = " mean execution time : %.4f sec. (%.1f FPS)\n max execution time : %.4f sec. (%.1f FPS)\n min execution time : %.4f sec. (%.1f FPS)" \
               % (float(mean_execute_time), float(mean_fps), float(max_execute_time), float(min_fps),
                  float(min_execute_time), float(max_fps))
    ax.text(0.55, 0.85, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    # fig.canvas.draw()
    # fig.canvas.flush_events()
    plt.show(block=False)
