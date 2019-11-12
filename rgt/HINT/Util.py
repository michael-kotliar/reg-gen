import os
import numpy as np
import pandas as pd
import pysam
import logomaker

import matplotlib.pyplot as plt

# Internal
from rgt.Util import AuxiliaryFunctions, GenomeData


def get_chromosome_size(organism):
    genome_data = GenomeData(organism)
    chrom_sizes_file_name = genome_data.get_chromosome_sizes()
    chrom_sizes_file = open(chrom_sizes_file_name, "r")
    chrom_sizes_dict = dict()
    for chrom_sizes_entry_line in chrom_sizes_file:
        chrom_sizes_entry_vec = chrom_sizes_entry_line.strip().split("\t")
        chrom_sizes_dict[chrom_sizes_entry_vec[0]] = int(chrom_sizes_entry_vec[1])
    chrom_sizes_file.close()

    return chrom_sizes_dict


def get_pwm(arguments):
    (organism, regions, window_size) = arguments
    pwm = dict([("A", [0.0] * window_size), ("C", [0.0] * window_size),
                ("G", [0.0] * window_size), ("T", [0.0] * window_size),
                ("N", [0.0] * window_size)])

    genome_data = GenomeData(organism)
    fasta = pysam.Fastafile(genome_data.get_genome())

    for region in regions:
        middle = (region.initial + region.final) / 2
        p1 = middle - window_size / 2
        p2 = middle + window_size / 2

        if p1 <= 0:
            continue

        aux_plus = 1
        dna_seq = str(fasta.fetch(region.chrom, p1, p2)).upper()

        if window_size % 2 == 0:
            aux_plus = 0

        dna_seq_rev = AuxiliaryFunctions.revcomp(str(fasta.fetch(region.chrom,
                                                                 p1 + aux_plus, p2 + aux_plus)).upper())
        if region.orientation == "+":
            for i in range(len(dna_seq)):
                pwm[dna_seq[i]][i] += 1

        elif region.orientation == "-":
            for i in range(len(dna_seq_rev)):
                pwm[dna_seq_rev[i]][i] += 1

    return pwm


def load_bias_table(table_file_name_f, table_file_name_r):
    bias_table_f, bias_table_r = dict(), dict()

    with open(table_file_name_f) as table_file_f:
        for line in table_file_f:
            ll = line.strip().split("\t")
            bias_table_f[ll[0]] = float(ll[1])

    with open(table_file_name_r) as table_file_r:
        for line in table_file_r:
            ll = line.strip().split("\t")
            bias_table_r[ll[0]] = float(ll[1])

    return [bias_table_f, bias_table_r]


def output_line_plot(arguments):
    (mpbs_name, mpbs_num, signal, pwm, output_location, window_size) = arguments
    mpbs_name = mpbs_name.replace("(", "_").replace(")", "")

    # output signal
    output_filename = os.path.join(output_location, "{}.txt".format(mpbs_name))
    with open(output_filename, "w") as f:
        f.write("\t".join(map(str, signal)) + "\n")

    # to create a motif loge, we only use A, C, G, T
    pwm = {k: pwm[k] for k in ('A', 'C', 'G', 'T')}
    pwm = pd.DataFrame(data=pwm)
    pwm = pwm.add(1)
    pwm_prob = (pwm.T / pwm.T.sum()).T
    pwm_prob_log = np.log2(pwm_prob)
    pwm_prob_log = pwm_prob_log.mul(pwm_prob)
    info_content = pwm_prob_log.T.sum() + 2
    icm = pwm_prob.mul(info_content, axis=0)

    start = int(-(window_size / 2))
    end = int((window_size / 2) - 1)
    x = np.linspace(start, end, num=window_size)

    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot(x, signal, color="red")

    ax.text(0.15, 0.9, 'n = {}'.format(mpbs_num), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontweight='bold')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 15))
    ax.tick_params(direction='out')
    ax.set_xticks([start, 0, end])
    ax.set_xticklabels([str(start), 0, str(end)])
    min_signal = np.min(signal)
    max_signal = np.max(signal)
    ax.set_yticks([min_signal, max_signal])
    ax.set_yticklabels([str(round(min_signal, 2)), str(round(max_signal, 2))], rotation=90)

    ax.set_title(mpbs_name, fontweight='bold')
    ax.set_xlim(start, end)
    ax.set_ylim([min_signal, max_signal])
    ax.spines['bottom'].set_position(('outward', 70))

    ax = plt.axes([0.105, 0.085, 0.85, .2])
    logo = logomaker.Logo(icm, ax=ax, show_spines=False, baseline_width=0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    output_filename = os.path.join(output_location, "{}.pdf".format(mpbs_name))
    plt.savefig(output_filename)


def output_line_plot_strand(arguments):
    (mpbs_name, mpbs_num, signal_forward, signal_reverse, pwm, output_location, window_size) = arguments
    mpbs_name = mpbs_name.replace("(", "_").replace(")", "")

    # output signal
    output_filename = os.path.join(output_location, "{}.txt".format(mpbs_name))
    with open(output_filename, "w") as f:
        f.write("\t".join(map(str, signal_forward)) + "\n")
        f.write("\t".join(map(str, signal_reverse)) + "\n")

    # to create a motif loge, we only use A, C, G, T
    pwm = {k: pwm[k] for k in ('A', 'C', 'G', 'T')}
    pwm = pd.DataFrame(data=pwm)
    pwm = pwm.add(1)
    pwm_prob = (pwm.T / pwm.T.sum()).T
    pwm_prob_log = np.log2(pwm_prob)
    pwm_prob_log = pwm_prob_log.mul(pwm_prob)
    info_content = pwm_prob_log.T.sum() + 2
    icm = pwm_prob.mul(info_content, axis=0)

    start = int(-(window_size / 2))
    end = int((window_size / 2) - 1)
    x = np.linspace(start, end, num=window_size)

    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot(x, signal_forward, color="red", label="Forward")
    ax.plot(x, signal_reverse, color="blue", label="Reverse")

    ax.text(0.15, 0.9, 'n = {}'.format(mpbs_num), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontweight='bold')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 15))
    ax.tick_params(direction='out')
    ax.set_xticks([start, 0, end])
    ax.set_xticklabels([str(start), 0, str(end)])
    min_signal = np.min([np.min(signal_forward), np.min(signal_reverse)])
    max_signal = np.max([np.max(signal_forward), np.max(signal_reverse)])
    ax.set_yticks([min_signal, max_signal])
    ax.set_yticklabels([str(round(min_signal, 2)), str(round(max_signal, 2))], rotation=90)

    ax.set_title(mpbs_name, fontweight='bold')
    ax.set_xlim(start, end)
    ax.set_ylim([min_signal, max_signal])
    ax.legend(loc="upper right", frameon=False)
    ax.spines['bottom'].set_position(('outward', 70))

    ax = plt.axes([0.105, 0.085, 0.85, .2])
    logo = logomaker.Logo(icm, ax=ax, show_spines=False, baseline_width=0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    output_filename = os.path.join(output_location, "{}.pdf".format(mpbs_name))
    plt.savefig(output_filename)


def output_line_plot_multi_conditions(arguments):
    (mpbs_name, mpbs_num, signals, conditions, pwm, output_location, window_size, colors) = arguments
    mpbs_name = mpbs_name.replace("(", "_").replace(")", "")

    # output signal
    output_filename = os.path.join(output_location, "{}.txt".format(mpbs_name))
    with open(output_filename, "w") as f:
        f.write("\t".join(conditions) + "\n")
        for i in range(window_size):
            res = []
            for j, condition in enumerate(conditions):
                res.append(signals[j][i])

            f.write("\t".join(map(str, res)) + "\n")

    # to create a motif loge, we only use A, C, G, T
    pwm = {k: pwm[k] for k in ('A', 'C', 'G', 'T')}
    pwm = pd.DataFrame(data=pwm)
    pwm = pwm.add(1)
    pwm_prob = (pwm.T / pwm.T.sum()).T
    pwm_prob_log = np.log2(pwm_prob)
    pwm_prob_log = pwm_prob_log.mul(pwm_prob)
    info_content = pwm_prob_log.T.sum() + 2
    icm = pwm_prob.mul(info_content, axis=0)

    start = int(-(window_size / 2))
    end = int((window_size / 2) - 1)
    x = np.linspace(start, end, num=window_size)

    plt.close('all')
    fig, ax = plt.subplots()
    for i, condition in enumerate(conditions):
        ax.plot(x, signals[i], color=colors[i], label=condition)

    ax.text(0.15, 0.9, 'n = {}'.format(mpbs_num), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontweight='bold')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 15))
    ax.tick_params(direction='out')
    ax.set_xticks([start, 0, end])
    ax.set_xticklabels([str(start), 0, str(end)])
    min_signal = np.min(signals)
    max_signal = np.max(signals)
    ax.set_yticks([min_signal, max_signal])
    ax.set_yticklabels([str(round(min_signal, 2)), str(round(max_signal, 2))], rotation=90)

    ax.set_title(mpbs_name, fontweight='bold')
    ax.set_xlim(start, end)
    ax.set_ylim([min_signal, max_signal])
    ax.legend(loc="upper right", frameon=False)
    ax.spines['bottom'].set_position(('outward', 70))

    ax = plt.axes([0.105, 0.085, 0.85, .2])
    logo = logomaker.Logo(icm, ax=ax, show_spines=False, baseline_width=0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    output_filename = os.path.join(output_location, "{}.pdf".format(mpbs_name))
    plt.savefig(output_filename)
