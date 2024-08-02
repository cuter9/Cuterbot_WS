# %matplotlib inline
# from IPython.display import clear_output

import numpy as np
import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# dir_training_records = os.path.join(dir_depo, 'training records', TRAIN_MODEL)
# os.makedirs(dir_training_records, exist_ok=True)

fig_1, ax_1 = plt.subplots(figsize=(14, 6))

font = {'fontweight': 'normal', 'fontsize': 16}
font_title = {'fontweight': 'medium', 'fontsize': 20}


# plot the training convergence profile
def plot_loss(loss_data, best_loss, no_epoch,
              dir_training_records, train_model, train_method, processor,
              show_training_plot=False):
    plt.cla()
    plt.tick_params(axis='both', labelsize='large')
    epochs = range(len(loss_data))
    ld_train = [ld[0] for ld in loss_data]
    ld_test = [ld[1] for ld in loss_data]
    ax_1.semilogy(epochs, ld_train, "r-", linewidth=2.0, label="Training Loss: {:.4E}".format(ld_train[-1]))
    ax_1.semilogy(epochs, ld_test, 'bs--', linewidth=2.0, label="Test Loss: {:.4E}".format(ld_test[-1]))
    ax_1.set_xlim(0, int(epochs[-1] * 1.1) + 1)
    xlim = epochs[-1] + 2
    ax_1.set_xlim(0, xlim)

    plt.title("Training convergence ({:s} with {:s}) -- {:s} \n current best test loss : {:.4f}".
              format(processor, train_method, train_model, best_loss),
              fontdict=font_title)
    plt.xlabel('epoch', fontdict=font)
    plt.ylabel('loss', fontdict=font)
    plt.legend(fontsize='x-large')

    fig_1.canvas.draw()
    fig_1.canvas.flush_events()
    if show_training_plot:
        plt.show(block=False)
    if len(loss_data) >= no_epoch:
        profile_plot = os.path.join(dir_training_records,
                                    "Training_convergence_plot_Model_{:s}_Training_Method_{:s})".
                                    format(train_model, train_method))
        fig_1.savefig(profile_plot)
    # plt.clf()


# plot the statistical histogram of learning time in terms of epoch and sample
def lt_plot(lt_epoch, lt_sample, overall_time, dir_training_records, train_model, train_method, processor):
    from math import ceil, floor
    import time
    # ----- training time statistics in terms of epoch
    # lt_epoch[0] -= lt_sample[0]
    learning_time_epoch = np.array(lt_epoch)
    mean_lt_epoch = np.mean(learning_time_epoch)
    max_lt_epoch = np.amax(learning_time_epoch)
    min_lt_epoch = np.amin(learning_time_epoch)
    print(
        "mean learning time per epoch: {:.3f} s, maximum epoch learning time: {:.3f} s, minimum epoch learning time: {:.3f} s".
        format(mean_lt_epoch, max_lt_epoch, min_lt_epoch))

    # ----- training time statistics in terms of sample
    lt_sample.sort(reverse=True)
    nex = ceil(0.001 * len(lt_sample))
    learning_time_sample = np.array(lt_sample[nex: -nex])
    mean_lt_sample = np.mean(learning_time_sample)
    max_lt_sample = np.amax(learning_time_sample)
    min_lt_sample = np.amin(learning_time_sample)
    print(
        "mean learning time per sample: {:.3f} s, maximum sample learning time: {:.3f} s, minimum sample learning time: {:.3f} s".
        format(mean_lt_sample, max_lt_sample, min_lt_sample))

    fig_2, axh = plt.subplots(1, 2, figsize=(14, 6))
    fig_2.suptitle("Training Time Statistics ({:s} with {:s}) -- {:s} \n Overall training time : {:s} ({:.2f} sec.)".
                   format(processor, train_method, train_model,
                          time.strftime("%H:%M:%S", time.gmtime(ceil(overall_time))), overall_time),
                   fontsize=20, fontweight='medium')
    axh[0].set_ylabel('no. of epoch', fontdict=font)
    axh[0].set_xlabel('time of training in an epoch, sec.', fontdict=font)
    cf = 0.9 * min_lt_epoch
    cc = 1.1 * max_lt_epoch
    bins_epochs_time = np.arange(cf, cc, (cc - cf) / 30)
    axh[0].hist(learning_time_epoch, bins=bins_epochs_time.tolist())
    axh[0].tick_params(axis='both', labelsize='large')
    props = dict(boxstyle='round', facecolor='wheat')
    text_str_0 = (" mean time: %.4f sec. \n max time: %.4f sec. \n min time: %.4f sec. "
                  % (float(mean_lt_epoch), float(max_lt_epoch), float(min_lt_epoch)))
    axh[0].text(0.55, 0.85, text_str_0, transform=axh[0].transAxes, fontsize=10, verticalalignment='top', bbox=props)

    axh[1].set_ylabel('no. of sample', fontdict=font)
    axh[1].set_xlabel('time for training a batch of samples , sec.', fontdict=font)
    sf = 0.9 * min_lt_sample
    sc = 1.1 * max_lt_sample
    bins_samples_time = np.arange(sf, sc, (sc - sf) / 30)
    axh[1].hist(learning_time_sample, bins=bins_samples_time.tolist())
    # axh[1].hist(learning_time_sample, bins=(0.01 * np.array(list(range(101)))).tolist())
    axh[1].tick_params(axis='both', labelsize='large')
    props = dict(boxstyle='round', facecolor='wheat')
    text_str_1 = (" mean time: %.4f sec. \n max time: %.4f sec. \n min time: %.4f sec. "
                  % (float(mean_lt_sample), float(max_lt_sample), float(min_lt_sample)))
    axh[1].text(0.55, 0.85, text_str_1, transform=axh[1].transAxes, fontsize=10, verticalalignment='top', bbox=props)

    fig_2.canvas.draw()
    fig_2.canvas.flush_events()
    plt.show(block=False)
    training_time_file = os.path.join(dir_training_records,
                                      "Training_time_Model_{:s}_Training_Method_{:s})".
                                      format(train_model, train_method))
    fig_2.savefig(training_time_file)
    # plt.clf()
