import os
import numpy as np
import pysam
from Bio import motifs
from scipy.signal import savgol_filter
from scipy.stats import scoreatpercentile
from argparse import SUPPRESS
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Internal
from rgt.Util import GenomeData, AuxiliaryFunctions, ErrorHandler
from rgt.HINT.GenomicSignal import GenomicSignal
from rgt.GenomicRegionSet import GenomicRegionSet
from rgt.HINT.BiasTable import BiasTable
from rgt.HINT.Util import *


def plotting_args(parser):
    # Parameters Options
    parser.add_argument("--organism", type=str, metavar="STRING", default="hg19",
                        help=("Organism considered on the analysis. Check our full documentation for all available "
                              "options. All default files such as genomes will be based on the chosen organism "
                              "and the data.config file."))
    parser.add_argument("--bias-table", type=str, metavar="FILE1_F,FILE1_R", default=None)
    parser.add_argument("--window-size", type=int, metavar="INT", default=400)
    parser.add_argument("--nc", type=int, metavar="INT", default=1,
                        help="The number of cores. DEFAULT: 1")

    # Hidden Options
    parser.add_argument("--initial-clip", type=int, metavar="INT", default=50, help=SUPPRESS)
    parser.add_argument("--forward-shift", type=int, metavar="INT", default=4, help=SUPPRESS)
    parser.add_argument("--reverse-shift", type=int, metavar="INT", default=-5, help=SUPPRESS)
    parser.add_argument("--k-nb", type=int, metavar="INT", default=6, help=SUPPRESS)
    parser.add_argument("--y-lim", type=float, metavar="FLOAT", default=0.3, help=SUPPRESS)

    # Output Options
    parser.add_argument("--output-location", type=str, metavar="PATH", default=os.getcwd(),
                        help="Path where the output bias table files will be written.")
    parser.add_argument("--output-prefix", type=str, metavar="STRING", default=None,
                        help="The prefix for results files.")

    # plot type
    parser.add_argument("--raw", default=False, action='store_true')
    parser.add_argument("--strand-specific", default=False, action='store_true')
    parser.add_argument("--bias-raw-bc-line", default=False, action='store_true')
    parser.add_argument("--raw-bc-line", default=False, action='store_true')
    parser.add_argument("--bias-line", default=False, action='store_true')
    parser.add_argument("--atac-dnase-line", default=False, action='store_true')
    parser.add_argument("--bias-raw-bc-strand-line2", default=False, action='store_true')
    parser.add_argument("--fragment-raw-size-line", default=False, action='store_true')
    parser.add_argument("--fragment-bc-size-line", default=False, action='store_true')

    parser.add_argument('input_files', metavar='reads.bam regions.bed', type=str, nargs='*',
                        help='BAM file of reads and BED files of interesting regions')


def plotting_run(args):
    if args.raw:
        if args.strand_specific:
            line_raw_signal_strand(args=args)
        else:
            line_raw_signal(args=args)

    if args.bias_raw_bc_line:
        bias_raw_bc_strand_line(args)

    if args.raw_bc_line:
        raw_bc_line(args)

    if args.bias_raw_bc_strand_line2:
        bias_raw_bc_strand_line2(args)

    if args.fragment_raw_size_line:
        fragment_size_raw_line(args)

    if args.fragment_bc_size_line:
        fragment_size_bc_line(args)


def get_data_for_region(args):
    # Initializing Error Handler
    err = ErrorHandler()

    if len(args.input_files) != 2:
        err.throw_error("ME_FEW_ARG", add_msg="You must specify reads and regions file.")

    bam_file, region_file = args.input_files[0], args.input_files[1]
    # check if index exists for bam file
    bam_index_file = "{}.bai".format(bam_file)
    if not os.path.exists(bam_index_file):
        pysam.index(bam_file)

    regions = GenomicRegionSet("Interested regions")
    regions.read(region_file)

    regions.sort()
    regions.remove_duplicates()
    region_name_list = list(set(regions.get_names()))

    region_len = list()
    region_num = list()
    region_pwm = list()

    print("{}: preparing data for plotting...\n".format(time.strftime("%D-%H:%M:%S")))
    mpbs_regions_by_name = dict()
    for region in regions:
        if region.name not in mpbs_regions_by_name:
            mpbs_regions_by_name[region.name] = GenomicRegionSet(region.name)
        mpbs_regions_by_name[region.name].add(region)

    for mpbs_name in region_name_list:
        region_len.append(mpbs_regions_by_name[mpbs_name][0].final - mpbs_regions_by_name[mpbs_name][0].initial)
        region_num.append(len(mpbs_regions_by_name[mpbs_name]))

    print("{}: generating pwm for each factor...\n".format(time.strftime("%D-%H:%M:%S")))
    if args.nc == 1:
        for region_name in region_name_list:
            pwm = get_pwm([args.organism, mpbs_regions_by_name[region_name], args.window_size])
            region_pwm.append(pwm)
    else:
        with Pool(processes=args.nc) as pool:
            arguments_list = list()
            for mpbs_name in region_name_list:
                arguments_list.append([args.organism, mpbs_regions_by_name[mpbs_name], args.window_size])

            pwm_list = pool.map(get_pwm, arguments_list)
            for i, mpbs_name in enumerate(region_name_list):
                region_pwm.append(pwm_list[i])

    return bam_file, region_name_list, mpbs_regions_by_name, region_len, region_num, region_pwm


def get_raw_signal(arguments):
    (bam_file, regions, organism, window_size, forward_shift, reverse_shift) = arguments

    genomic_signal = GenomicSignal(bam_file)

    signal = np.zeros(window_size)
    for region in regions:
        mid = (region.final + region.initial) / 2
        p1 = int(mid - window_size / 2)
        p2 = int(mid + window_size / 2)
        if p1 <= 0:
            continue

        signal += genomic_signal.get_raw_signal(chromosome=region.chrom,
                                                start=p1,
                                                end=p2,
                                                forward_shift=forward_shift,
                                                reverse_shift=reverse_shift,
                                                strand_specific=False)

    signal /= len(regions)
    return signal


def get_bc_signal(arguments):
    (bam_file, regions, organism, window_size, forward_shift, reverse_shift) = arguments

    genomic_signal = GenomicSignal(bam_file)

    signal = np.zeros(window_size)
    for region in regions:
        mid = (region.final + region.initial) / 2
        p1 = int(mid - window_size / 2)
        p2 = int(mid + window_size / 2)
        if p1 <= 0:
            continue

        signal += genomic_signal.get_raw_signal(chromosome=region.chrom,
                                                start=p1,
                                                end=p2,
                                                forward_shift=forward_shift,
                                                reverse_shift=reverse_shift,
                                                strand_specific=False)

    signal /= len(regions)
    return signal


def get_raw_signal_strand(arguments):
    (bam_file, regions, organism, window_size, forward_shift, reverse_shift) = arguments

    genomic_signal = GenomicSignal(bam_file)

    signal_forward, signal_reverse = np.zeros(window_size), np.zeros(window_size)
    for region in regions:
        mid = (region.final + region.initial) / 2
        p1 = int(mid - window_size / 2)
        p2 = int(mid + window_size / 2)
        if p1 <= 0:
            continue

        signal_f, signal_r = genomic_signal.get_raw_signal(chromosome=region.chrom,
                                                           start=p1,
                                                           end=p2,
                                                           forward_shift=forward_shift,
                                                           reverse_shift=reverse_shift,
                                                           strand_specific=True)

        signal_forward += signal_f
        signal_reverse += signal_r

    signal_forward /= len(regions)
    signal_reverse /= len(regions)

    return signal_forward, signal_reverse


def line_raw_signal(args):
    bam_file, region_name_list, mpbs_regions_by_name, region_len, region_num, region_pwm = get_data_for_region(args)

    print("{}: generating signal for each factor...\n".format(time.strftime("%D-%H:%M:%S")))

    signals = np.zeros(shape=(len(region_name_list), args.window_size), dtype=np.float32)
    if args.nc == 1:
        for i, region_name in enumerate(region_name_list):
            regions = mpbs_regions_by_name[region_name]
            arguments = (bam_file, regions, args.organism, args.window_size, args.forward_shift,
                         args.reverse_shift)

            signals[i] = get_raw_signal(arguments)
    else:
        with Pool(processes=args.nc) as pool:
            arguments_list = list()
            for i, region_name in enumerate(region_name_list):
                regions = mpbs_regions_by_name[region_name]
                arguments_list.append([bam_file, regions, args.organism, args.window_size, args.forward_shift,
                                       args.reverse_shift])

            signal_list = pool.map(get_raw_signal, arguments_list)
            for i, region_name in enumerate(region_name_list):
                signals[i] = signal_list[i]

    print("{}: generating plot for each factor...\n".format(time.strftime("%D-%H:%M:%S")))

    if args.nc == 1:
        for i, region_name in enumerate(region_name_list):
            arguments = (region_name, region_num[i], signals[i], region_pwm[i], args.output_location,
                         args.window_size)
            output_line_plot(arguments)
    else:
        with Pool(processes=args.nc) as pool:
            arguments_list = list()
            for i, region_name in enumerate(region_name_list):
                arguments_list.append([region_name, region_num[i], signals[i], region_pwm[i], args.output_location,
                                       args.window_size])

            pool.map(output_line_plot, arguments_list)


def line_raw_signal_strand(args):
    bam_file, region_name_list, mpbs_regions_by_name, region_len, region_num, region_pwm = get_data_for_region(args)

    print("{}: generating signal for each factor...\n".format(time.strftime("%D-%H:%M:%S")))

    signals_forward = np.zeros(shape=(len(region_name_list), args.window_size), dtype=np.float32)
    signals_reverse = np.zeros(shape=(len(region_name_list), args.window_size), dtype=np.float32)
    if args.nc == 1:
        for i, region_name in enumerate(region_name_list):
            regions = mpbs_regions_by_name[region_name]
            arguments = (bam_file, regions, args.organism, args.window_size, args.forward_shift,
                         args.reverse_shift)

            signals_forward[i], signals_reverse[i] = get_raw_signal_strand(arguments)
    else:
        with Pool(processes=args.nc) as pool:
            arguments_list = list()
            for i, region_name in enumerate(region_name_list):
                regions = mpbs_regions_by_name[region_name]
                arguments_list.append([bam_file, regions, args.organism, args.window_size, args.forward_shift,
                                       args.reverse_shift])

            signal_list = pool.map(get_raw_signal, arguments_list)
            for i, region_name in enumerate(region_name_list):
                signals_forward[i] = signal_list[i][0]
                signals_reverse[i] = signal_list[i][0]

    print("{}: generating plot for each factor...\n".format(time.strftime("%D-%H:%M:%S")))

    if args.nc == 1:
        for i, region_name in enumerate(region_name_list):
            arguments = (region_name, region_num[i], signals_forward[i], signals_reverse[i], region_pwm[i],
                         args.output_location, args.window_size)
            output_line_plot_strand(arguments)
    else:
        with Pool(processes=args.nc) as pool:
            arguments_list = list()
            for i, region_name in enumerate(region_name_list):
                arguments_list.append([region_name, region_num[i], signals_forward[i], signals_reverse[i],
                                       region_pwm[i], args.output_location, args.window_size])

            pool.map(output_line_plot_strand, arguments_list)


def line_bc_signal(args):
    bam_file, region_name_list, mpbs_regions_by_name, region_len, region_num, region_pwm = get_data_for_region(args)


def bias_raw_bc_line(args):
    signal = GenomicSignal(args.reads_file)
    signal.load_sg_coefs(slope_window_size=9)
    bias_table = BiasTable()
    bias_table_list = args.bias_table.split(",")
    table = bias_table.load_table(table_file_name_F=bias_table_list[0],
                                  table_file_name_R=bias_table_list[1])

    genome_data = GenomeData(args.organism)
    fasta = pysam.Fastafile(genome_data.get_genome())
    pwm_dict = dict([("A", [0.0] * args.window_size), ("C", [0.0] * args.window_size),
                     ("G", [0.0] * args.window_size), ("T", [0.0] * args.window_size),
                     ("N", [0.0] * args.window_size)])

    num_sites = 0

    mpbs_regions = GenomicRegionSet("Motif Predicted Binding Sites")
    mpbs_regions.read(args.motif_file)

    bam = Samfile(args.reads_file, "rb")

    mean_signal_bias_f = np.zeros(args.window_size)
    mean_signal_bias_r = np.zeros(args.window_size)
    mean_signal_raw = np.zeros(args.window_size)
    mean_signal_raw_f = np.zeros(args.window_size)
    mean_signal_raw_r = np.zeros(args.window_size)
    mean_signal_bc = np.zeros(args.window_size)
    mean_signal_bc_f = np.zeros(args.window_size)
    mean_signal_bc_r = np.zeros(args.window_size)

    motif_len = 0

    for region in mpbs_regions:
        if str(region.name).split(":")[-1] == "Y":
            # Extend by window_size
            mid = (region.initial + region.final) / 2
            p1 = mid - (args.window_size / 2)
            p2 = mid + (args.window_size / 2)
            motif_len = region.final - region.initial

            signal_bias_f, signal_bias_r, raw, raw_f, raw_r, bc, bc_f, bc_r = \
                signal.get_bias_raw_bc_signal(ref=region.chrom, start=p1, end=p2, bam=bam,
                                              fasta=fasta, bias_table=table,
                                              forward_shift=args.forward_shift,
                                              reverse_shift=args.reverse_shift,
                                              strand=True)

            num_sites += 1
            mean_signal_bias_f = np.add(mean_signal_bias_f, np.array(signal_bias_f))
            mean_signal_bias_r = np.add(mean_signal_bias_r, np.array(signal_bias_r))
            mean_signal_raw = np.add(mean_signal_raw, np.array(raw))
            mean_signal_raw_f = np.add(mean_signal_raw_f, np.array(raw_f))
            mean_signal_raw_r = np.add(mean_signal_raw_r, np.array(raw_r))
            mean_signal_bc = np.add(mean_signal_bc, np.array(bc))
            mean_signal_bc_f = np.add(mean_signal_bc_f, np.array(bc_f))
            mean_signal_bc_r = np.add(mean_signal_bc_r, np.array(bc_r))

            # Update pwm
            aux_plus = 1
            dna_seq = str(fasta.fetch(region.chrom, p1, p2)).upper()
            if (region.final - region.initial) % 2 == 0:
                aux_plus = 0

            if region.orientation == "+":
                for i in range(len(dna_seq)):
                    pwm_dict[dna_seq[i]][i] += 1

    mean_signal_bias_f = mean_signal_bias_f / num_sites
    mean_signal_bias_r = mean_signal_bias_r / num_sites
    mean_signal_raw = mean_signal_raw / num_sites
    mean_signal_bc = mean_signal_bc / num_sites
    mean_signal_bc_f = mean_signal_bc_f / num_sites
    mean_signal_bc_r = mean_signal_bc_r / num_sites

    # Output the norm and slope signal
    output_fname = os.path.join(args.output_location, "{}.txt".format(args.output_prefix))
    f = open(output_fname, "w")
    f.write("\t".join((list(map(str, mean_signal_bias_f)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bias_r)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_raw)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bc)))) + "\n")
    f.close()

    # Output PWM and create logo
    pwm_fname = os.path.join(args.output_location, "{}.pwm".format(args.output_prefix))
    pwm_file = open(pwm_fname, "w")
    for e in ["A", "C", "G", "T"]:
        pwm_file.write(" ".join([str(int(f)) for f in pwm_dict[e]]) + "\n")
    pwm_file.close()

    logo_fname = os.path.join(args.output_location, "{}.logo.eps".format(args.output_prefix))
    pwm = motifs.read(open(pwm_fname), "pfm")

    pwm.weblogo(logo_fname, format="eps", stack_width="large", stacks_per_line=str(args.window_size),
                color_scheme="color_classic", unit_name="", show_errorbars=False, logo_title="",
                show_xaxis=False, xaxis_label="", show_yaxis=False, yaxis_label="",
                show_fineprint=False, show_ends=False)

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 6))

    start = -(args.window_size / 2)
    end = (args.window_size / 2) - 1
    x = np.linspace(start, end, num=args.window_size)

    if motif_len % 2 == 0:
        x1 = int(- (motif_len / 2))
        x2 = int(motif_len / 2)
    else:
        x1 = int(-(motif_len / 2) - 1)
        x2 = int((motif_len / 2) + 1)

    ############################################################
    # bias signal per strand
    fp_score = sum(mean_signal_raw[args.window_size / 2 + x1: args.window_size / 2 + x2])
    shoulder_l = sum(mean_signal_raw[args.window_size / 2 + x1 - motif_len:args.window_size / 2 + x1])
    shoulder_r = sum(mean_signal_raw[args.window_size / 2 + x2:args.window_size / 2 + x2 + motif_len])
    sfr = (shoulder_l + shoulder_r) / (2 * fp_score)
    min_ax1 = min(mean_signal_raw)
    max_ax1 = max(mean_signal_raw)
    ax1.plot(x, mean_signal_raw, color='blue', label='Uncorrected')
    ax1.text(0.15, 0.9, 'n = {}'.format(num_sites), verticalalignment='bottom',
             horizontalalignment='right', transform=ax1.transAxes, fontweight='bold')
    ax1.text(0.35, 0.15, 'SFR = {}'.format(round(sfr, 2)), verticalalignment='bottom',
             horizontalalignment='right', transform=ax1.transAxes, fontweight='bold')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_position(('outward', 15))
    ax1.spines['bottom'].set_position(('outward', 5))
    ax1.tick_params(direction='out')
    ax1.set_xticks([start, 0, end])
    ax1.set_xticklabels([str(start), 0, str(end)])
    ax1.set_yticks([min_ax1, max_ax1])
    ax1.set_yticklabels([str(round(min_ax1, 2)), str(round(max_ax1, 2))], rotation=90)
    ax1.set_title(args.output_prefix, fontweight='bold')
    ax1.set_xlim(start, end)
    ax1.set_ylim([min_ax1, max_ax1])
    ax1.legend(loc="lower right", frameon=False)
    ####################################################################

    #####################################################################
    # Bias corrected, non-bias corrected (not strand specific)
    fp_score = sum(mean_signal_bc[args.window_size / 2 + x1: args.window_size / 2 + x2])
    shoulder_l = sum(mean_signal_bc[args.window_size / 2 + x1 - motif_len:args.window_size / 2 + x1])
    shoulder_r = sum(mean_signal_bc[args.window_size / 2 + x2:args.window_size / 2 + x2 + motif_len])
    sfr = (shoulder_l + shoulder_r) / (2 * fp_score)
    min_ax2 = min(mean_signal_bc)
    max_ax2 = max(mean_signal_bc)
    ax2.plot(x, mean_signal_bc, color='red', label='Corrected')
    ax2.text(0.35, 0.15, 'SFR = {}'.format(round(sfr, 2)), verticalalignment='bottom',
             horizontalalignment='right', transform=ax2.transAxes, fontweight='bold')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_position(('outward', 15))
    ax2.tick_params(direction='out')
    ax2.set_xticks([start, 0, end])
    ax2.set_xticklabels([str(start), 0, str(end)])
    ax2.set_yticks([min_ax2, max_ax2])
    ax2.set_yticklabels([str(round(min_ax2, 2)), str(round(max_ax2, 2))], rotation=90)
    ax2.set_xlim(start, end)
    ax2.set_ylim([min_ax2, max_ax2])
    ax2.legend(loc="lower right", frameon=False)

    fp_score_f = sum(mean_signal_bc_f[args.window_size / 2 + x1: args.window_size / 2 + x2])
    shoulder_l_f = sum(mean_signal_bc_f[args.window_size / 2 + x1 - motif_len:args.window_size / 2 + x1])
    shoulder_r_f = sum(mean_signal_bc_f[args.window_size / 2 + x2:args.window_size / 2 + x2 + motif_len])
    sfr_f = (shoulder_l_f + shoulder_r_f) / (2 * fp_score_f)
    fp_score_r = sum(mean_signal_bc_r[args.window_size / 2 + x1: args.window_size / 2 + x2])
    shoulder_l_r = sum(mean_signal_bc_r[args.window_size / 2 + x1 - motif_len:args.window_size / 2 + x1])
    shoulder_r_r = sum(mean_signal_bc_r[args.window_size / 2 + x2:args.window_size / 2 + x2 + motif_len])
    sfr_r = (shoulder_l_r + shoulder_r_r) / (2 * fp_score_r)
    min_ax3 = min(min(mean_signal_bc_f), min(mean_signal_bc_r))
    max_ax3 = max(max(mean_signal_bc_f), max(mean_signal_bc_r))
    ax3.plot(x, mean_signal_bc_f, color='purple', label='Forward')
    ax3.plot(x, mean_signal_bc_r, color='green', label='Reverse')
    ax3.text(0.35, 0.15, 'SFR_f = {}'.format(round(sfr_f, 2)), verticalalignment='bottom',
             horizontalalignment='right', transform=ax3.transAxes, fontweight='bold')
    ax3.text(0.35, 0.05, 'SFR_r = {}'.format(round(sfr_r, 2)), verticalalignment='bottom',
             horizontalalignment='right', transform=ax3.transAxes, fontweight='bold')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_position(('outward', 15))
    ax3.tick_params(direction='out')
    ax3.set_xticks([start, 0, end])
    ax3.set_xticklabels([str(start), 0, str(end)])
    ax3.set_yticks([min_ax3, max_ax3])
    ax3.set_yticklabels([str(round(min_ax3, 2)), str(round(max_ax3, 2))], rotation=90)
    ax3.set_xlim(start, end)
    ax3.set_ylim([min_ax3, max_ax3])
    ax3.legend(loc="lower right", frameon=False)

    ax3.spines['bottom'].set_position(('outward', 40))

    ax1.axvline(x=x1, ymin=-0.3, ymax=1, c="black", lw=0.5, ls='dashed', zorder=0, clip_on=False)
    ax1.axvline(x=x2, ymin=-0.3, ymax=1, c="black", lw=0.5, ls='dashed', zorder=0, clip_on=False)
    ax2.axvline(x=x1, ymin=-0.5, ymax=1.2, c="black", lw=0.5, ls='dashed', zorder=0, clip_on=False)
    ax2.axvline(x=x2, ymin=-0.5, ymax=1.2, c="black", lw=0.5, ls='dashed', zorder=0, clip_on=False)
    ###############################################################################
    # merge the above figures
    figure_name = os.path.join(args.output_location, "{}.line.eps".format(args.output_prefix))
    fig.subplots_adjust(bottom=.2, hspace=.5)
    fig.tight_layout()
    fig.savefig(figure_name, format="eps", dpi=300)

    # Creating canvas and printing eps / pdf with merged results
    output_fname = os.path.join(args.output_location, "{}.eps".format(args.output_prefix))
    c = pyx.canvas.canvas()
    c.insert(pyx.epsfile.epsfile(0, 0, figure_name, scale=1.0))
    c.insert(pyx.epsfile.epsfile(1.45, 0.89, logo_fname, width=18.3, height=1.75))
    c.writeEPSfile(output_fname)
    os.system("epstopdf " + figure_name)
    os.system("epstopdf " + logo_fname)
    os.system("epstopdf " + output_fname)

    os.remove(pwm_fname)
    os.remove(os.path.join(args.output_location, "{}.line.eps".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.logo.eps".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.line.pdf".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.logo.pdf".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.eps".format(args.output_prefix)))


def raw_bc_line(args):
    signal = GenomicSignal(args.reads_file)
    signal.load_sg_coefs(slope_window_size=9)
    bias_table = BiasTable()
    bias_table_list = args.bias_table.split(",")
    table = bias_table.load_table(table_file_name_F=bias_table_list[0],
                                  table_file_name_R=bias_table_list[1])

    genome_data = GenomeData(args.organism)
    fasta = Fastafile(genome_data.get_genome())
    pwm_dict = dict([("A", [0.0] * args.window_size), ("C", [0.0] * args.window_size),
                     ("G", [0.0] * args.window_size), ("T", [0.0] * args.window_size),
                     ("N", [0.0] * args.window_size)])

    num_sites = 0

    mpbs_regions = GenomicRegionSet("Motif Predicted Binding Sites")
    mpbs_regions.read(args.motif_file)

    bam = Samfile(args.reads_file, "rb")

    mean_signal_raw = np.zeros(args.window_size)
    mean_signal_bc = np.zeros(args.window_size)

    for region in mpbs_regions:
        if str(region.name).split(":")[-1] == "Y":
            # Extend by window_size
            mid = (region.initial + region.final) / 2
            p1 = mid - (args.window_size / 2)
            p2 = mid + (args.window_size / 2)
            signal_bias_f, signal_bias_r, raw, raw_f, raw_r, bc, bc_f, bc_r = \
                signal.get_bias_raw_bc_signal(ref=region.chrom, start=p1, end=p2, bam=bam,
                                              fasta=fasta, bias_table=table,
                                              forward_shift=args.forward_shift,
                                              reverse_shift=args.reverse_shift,
                                              strand=True)

            num_sites += 1
            mean_signal_raw = np.add(mean_signal_raw, np.array(raw))
            mean_signal_bc = np.add(mean_signal_bc, np.array(bc))

            # Update pwm
            aux_plus = 1
            dna_seq = str(fasta.fetch(region.chrom, p1, p2)).upper()
            if (region.final - region.initial) % 2 == 0:
                aux_plus = 0

            if region.orientation == "+":
                for i in range(len(dna_seq)):
                    pwm_dict[dna_seq[i]][i] += 1

    mean_signal_raw = mean_signal_raw / num_sites
    mean_signal_bc = mean_signal_bc / num_sites

    # Output the norm and slope signal
    output_fname = os.path.join(args.output_location, "{}.txt".format(args.output_prefix))
    f = open(output_fname, "w")
    f.write("\t".join((list(map(str, mean_signal_raw)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bc)))) + "\n")
    f.close()

    # Output PWM and create logo
    pwm_fname = os.path.join(args.output_location, "{}.pwm".format(args.output_prefix))
    pwm_file = open(pwm_fname, "w")
    for e in ["A", "C", "G", "T"]:
        pwm_file.write(" ".join([str(int(f)) for f in pwm_dict[e]]) + "\n")
    pwm_file.close()

    logo_fname = os.path.join(args.output_location, "{}.logo.eps".format(args.output_prefix))
    pwm = motifs.read(open(pwm_fname), "pfm")

    pwm.weblogo(logo_fname, format="eps", stack_width="large", stacks_per_line=str(args.window_size),
                color_scheme="color_classic", unit_name="", show_errorbars=False, logo_title="",
                show_xaxis=False, xaxis_label="", show_yaxis=False, yaxis_label="",
                show_fineprint=False, show_ends=False)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    start = -(args.window_size / 2)
    end = (args.window_size / 2) - 1
    x = np.linspace(start, end, num=args.window_size)

    ############################################################
    min_ = min(min(mean_signal_raw), min(mean_signal_bc))
    max_ = max(max(mean_signal_raw), max(mean_signal_bc))
    ax.plot(x, mean_signal_raw, color='red', label='Uncorrected')
    ax.plot(x, mean_signal_bc, color='blue', label='Corrected')
    ax.text(0.15, 0.9, 'n = {}'.format(num_sites), verticalalignment='bottom',
            horizontalalignment='right', transform=ax.transAxes, fontweight='bold')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 15))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.tick_params(direction='out')
    ax.set_xticks([start, 0, end])
    ax.set_xticklabels([str(start), 0, str(end)])
    ax.set_yticks([min_, max_])
    ax.set_yticklabels([str(round(min_, 2)), str(round(max_, 2))], rotation=90)
    ax.set_title(args.output_prefix, fontweight='bold')
    ax.set_xlim(start, end)
    ax.set_ylim(min_, max_)
    ax.legend(loc="lower right", frameon=False)

    ax.spines['bottom'].set_position(('outward', 40))

    ###############################################################################
    # merge the above figures
    figure_name = os.path.join(args.output_location, "{}.line.eps".format(args.output_prefix))
    fig.subplots_adjust(bottom=.2, hspace=.5)
    fig.tight_layout()
    fig.savefig(figure_name, format="eps", dpi=300)

    # Creating canvas and printing eps / pdf with merged results
    output_fname = os.path.join(args.output_location, "{}.eps".format(args.output_prefix))
    c = pyx.canvas.canvas()
    c.insert(pyx.epsfile.epsfile(0, 0, figure_name, scale=1.0))
    c.insert(pyx.epsfile.epsfile(1.45, 0.89, logo_fname, width=18.3, height=1.75))
    c.writeEPSfile(output_fname)
    os.system("epstopdf " + figure_name)
    os.system("epstopdf " + logo_fname)
    os.system("epstopdf " + output_fname)

    os.remove(pwm_fname)
    os.remove(os.path.join(args.output_location, "{}.line.eps".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.logo.eps".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.line.pdf".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.logo.pdf".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.eps".format(args.output_prefix)))


def bias_raw_bc_strand_line(args):
    signal = GenomicSignal(args.reads_file)
    signal.load_sg_coefs(slope_window_size=9)
    bias_table = BiasTable()
    bias_table_list = args.bias_table.split(",")
    table = bias_table.load_table(table_file_name_F=bias_table_list[0],
                                  table_file_name_R=bias_table_list[1])

    genome_data = GenomeData(args.organism)
    fasta = Fastafile(genome_data.get_genome())
    pwm_dict = dict([("A", [0.0] * args.window_size), ("C", [0.0] * args.window_size),
                     ("G", [0.0] * args.window_size), ("T", [0.0] * args.window_size),
                     ("N", [0.0] * args.window_size)])

    num_sites = 0

    mpbs_regions = GenomicRegionSet("Motif Predicted Binding Sites")
    mpbs_regions.read(args.motif_file)

    bam = Samfile(args.reads_file, "rb")

    mean_signal_bias_f = np.zeros(args.window_size)
    mean_signal_bias_r = np.zeros(args.window_size)
    mean_signal_raw = np.zeros(args.window_size)
    mean_signal_bc = np.zeros(args.window_size)
    for region in mpbs_regions:
        if str(region.name).split(":")[-1] == "Y":
            mid = (region.initial + region.final) / 2
            p1 = mid - (args.window_size / 2)
            p2 = mid + (args.window_size / 2)

            signal_bias_f, signal_bias_r, signal_raw, signal_bc = \
                signal.get_bias_raw_bc_signal(ref=region.chrom, start=p1, end=p2, bam=bam,
                                              fasta=fasta, bias_table=table,
                                              forward_shift=args.forward_shift, reverse_shift=args.reverse_shift)

            num_sites += 1
            mean_signal_bias_f = np.add(mean_signal_bias_f, np.array(signal_bias_f))
            mean_signal_bias_r = np.add(mean_signal_bias_r, np.array(signal_bias_r))
            mean_signal_raw = np.add(mean_signal_raw, np.array(signal_raw))
            mean_signal_bc = np.add(mean_signal_bc, np.array(signal_bc))

            # Update pwm
            aux_plus = 1
            dna_seq = str(fasta.fetch(region.chrom, p1, p2)).upper()
            if (region.final - region.initial) % 2 == 0:
                aux_plus = 0
            dna_seq_rev = AuxiliaryFunctions.revcomp(str(fasta.fetch(region.chrom,
                                                                     p1 + aux_plus, p2 + aux_plus)).upper())
            if region.orientation == "+":
                for i in range(0, len(dna_seq)):
                    pwm_dict[dna_seq[i]][i] += 1

    mean_signal_bias_f = mean_signal_bias_f / num_sites
    mean_signal_bias_r = mean_signal_bias_r / num_sites
    mean_signal_raw = mean_signal_raw / num_sites
    mean_signal_bc = mean_signal_bc / num_sites

    # Output the norm and slope signal
    output_fname = os.path.join(args.output_location, "{}.txt".format(args.output_prefix))
    f = open(output_fname, "w")
    f.write("\t".join((list(map(str, mean_signal_bias_f)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bias_r)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_raw)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bc)))) + "\n")
    f.close()

    # Output PWM and create logo
    pwm_fname = os.path.join(args.output_location, "{}.pwm".format(args.output_prefix))
    pwm_file = open(pwm_fname, "w")
    for e in ["A", "C", "G", "T"]:
        pwm_file.write(" ".join([str(int(f)) for f in pwm_dict[e]]) + "\n")
    pwm_file.close()

    logo_fname = os.path.join(args.output_location, "{}.logo.eps".format(args.output_prefix))
    pwm = motifs.read(open(pwm_fname), "pfm")

    pwm.weblogo(logo_fname, format="eps", stack_width="large", stacks_per_line=str(args.window_size),
                color_scheme="color_classic", unit_name="", show_errorbars=False, logo_title="",
                show_xaxis=False, xaxis_label="", show_yaxis=False, yaxis_label="",
                show_fineprint=False, show_ends=False)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 4))

    start = -(args.window_size / 2)
    end = (args.window_size / 2) - 1
    x = np.linspace(start, end, num=args.window_size)

    ############################################################
    # bias signal per strand
    min_ = min(min(mean_signal_bias_f), min(mean_signal_bias_r))
    max_ = max(max(mean_signal_bias_f), max(mean_signal_bias_r))
    ax1.plot(x, mean_signal_bias_f, color='purple', label='Forward')
    ax1.plot(x, mean_signal_bias_r, color='green', label='Reverse')
    ax1.text(0.15, 0.9, 'n = {}'.format(num_sites), verticalalignment='bottom',
             horizontalalignment='right', transform=ax1.transAxes, fontweight='bold')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_position(('outward', 15))
    ax1.spines['bottom'].set_position(('outward', 5))
    ax1.tick_params(direction='out')
    ax1.set_xticks([start, 0, end])
    ax1.set_xticklabels([str(start), 0, str(end)])
    ax1.set_yticks([min_, max_])
    ax1.set_yticklabels([str(round(min_, 2)), str(round(max_, 2))], rotation=90)
    ax1.set_title(args.output_prefix, fontweight='bold')
    ax1.set_xlim(start, end)
    ax1.set_ylim([min_, max_])
    ax1.legend(loc="upper right", frameon=False)
    ####################################################################

    #####################################################################
    # Bias corrected, non-bias corrected (not strand specific)
    min_ = min(min(mean_signal_raw), min(mean_signal_bc))
    max_ = max(max(mean_signal_raw), max(mean_signal_bc))
    ax2.plot(x, mean_signal_raw, color='blue', label='Uncorrected')
    ax2.plot(x, mean_signal_bc, color='red', label='Corrected')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_position(('outward', 15))
    ax2.tick_params(direction='out')
    ax2.set_xticks([start, 0, end])
    ax2.set_xticklabels([str(start), 0, str(end)])
    ax2.set_yticks([min_, max_])
    ax2.set_yticklabels([str(round(min_, 2)), str(round(max_, 2))], rotation=90)
    ax2.set_xlim(start, end)
    ax2.set_ylim([min_, max_])
    ax2.legend(loc="upper right", frameon=False)

    ax2.spines['bottom'].set_position(('outward', 40))

    ###############################################################################
    # merge the above figures
    figure_name = os.path.join(args.output_location, "{}.line.eps".format(args.output_prefix))
    fig.subplots_adjust(bottom=.2, hspace=.5)
    fig.tight_layout()
    fig.savefig(figure_name, format="eps", dpi=300)

    # Creating canvas and printing eps / pdf with merged results
    output_fname = os.path.join(args.output_location, "{}.eps".format(args.output_prefix))
    c = pyx.canvas.canvas()
    c.insert(pyx.epsfile.epsfile(0, 0, figure_name, scale=1.0))
    c.insert(pyx.epsfile.epsfile(1.51, 0.89, logo_fname, width=18.3, height=1.75))
    c.writeEPSfile(output_fname)
    os.system("epstopdf " + figure_name)
    os.system("epstopdf " + logo_fname)
    os.system("epstopdf " + output_fname)

    os.remove(pwm_fname)
    os.remove(os.path.join(args.output_location, "{}.line.eps".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.logo.eps".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.line.pdf".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.logo.pdf".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.eps".format(args.output_prefix)))


def bias_raw_bc_strand_line2(args):
    signal = GenomicSignal(args.reads_file)
    signal.load_sg_coefs(slope_window_size=9)
    bias_table = BiasTable()
    bias_table_list = args.bias_table.split(",")
    table = bias_table.load_table(table_file_name_F=bias_table_list[0],
                                  table_file_name_R=bias_table_list[1])

    genome_data = GenomeData(args.organism)
    fasta = Fastafile(genome_data.get_genome())
    pwm_dict = dict([("A", [0.0] * args.window_size), ("C", [0.0] * args.window_size),
                     ("G", [0.0] * args.window_size), ("T", [0.0] * args.window_size),
                     ("N", [0.0] * args.window_size)])

    num_sites = 0

    mpbs_regions = GenomicRegionSet("Motif Predicted Binding Sites")
    mpbs_regions.read(args.motif_file)

    bam = Samfile(args.reads_file, "rb")

    mean_signal_bias_f = np.zeros(args.window_size)
    mean_signal_bias_r = np.zeros(args.window_size)
    mean_signal_raw = np.zeros(args.window_size)
    mean_signal_raw_f = np.zeros(args.window_size)
    mean_signal_raw_r = np.zeros(args.window_size)
    mean_signal_bc = np.zeros(args.window_size)
    mean_signal_bc_f = np.zeros(args.window_size)
    mean_signal_bc_r = np.zeros(args.window_size)
    for region in mpbs_regions:
        if str(region.name).split(":")[-1] == "Y":
            mid = (region.initial + region.final) / 2
            p1 = mid - (args.window_size / 2)
            p2 = mid + (args.window_size / 2)

            signal_bias_f, signal_bias_r, signal_raw, signal_raw_f, signal_raw_r, signal_bc, signal_bc_f, signal_bc_r = \
                signal.get_bias_raw_bc_signal(ref=region.chrom, start=p1, end=p2, bam=bam,
                                              fasta=fasta, bias_table=table,
                                              forward_shift=args.forward_shift,
                                              reverse_shift=args.reverse_shift,
                                              strand=True)

            num_sites += 1
            mean_signal_bias_f = np.add(mean_signal_bias_f, np.array(signal_bias_f))
            mean_signal_bias_r = np.add(mean_signal_bias_r, np.array(signal_bias_r))
            mean_signal_raw = np.add(mean_signal_raw, np.array(signal_raw))
            mean_signal_raw_f = np.add(mean_signal_raw_f, np.array(signal_raw_f))
            mean_signal_raw_r = np.add(mean_signal_raw_r, np.array(signal_raw_r))
            mean_signal_bc = np.add(mean_signal_bc, np.array(signal_bc))
            mean_signal_bc_f = np.add(mean_signal_bc_f, np.array(signal_bc_f))
            mean_signal_bc_r = np.add(mean_signal_bc_r, np.array(signal_bc_r))

            # Update pwm
            aux_plus = 1
            dna_seq = str(fasta.fetch(region.chrom, p1, p2)).upper()
            if (region.final - region.initial) % 2 == 0:
                aux_plus = 0
            dna_seq_rev = AuxiliaryFunctions.revcomp(str(fasta.fetch(region.chrom,
                                                                     p1 + aux_plus, p2 + aux_plus)).upper())
            if region.orientation == "+":
                for i in range(0, len(dna_seq)):
                    pwm_dict[dna_seq[i]][i] += 1

    mean_signal_bias_f = mean_signal_bias_f / num_sites
    mean_signal_bias_r = mean_signal_bias_r / num_sites
    mean_signal_raw = mean_signal_raw / num_sites
    mean_signal_raw_f = mean_signal_raw_f / num_sites
    mean_signal_raw_r = mean_signal_raw_r / num_sites
    mean_signal_bc = mean_signal_bc / num_sites
    mean_signal_bc_f = mean_signal_bc_f / num_sites
    mean_signal_bc_r = mean_signal_bc_r / num_sites

    # mean_signal_raw = rescaling(mean_signal_raw)
    # mean_signal_bc = rescaling(mean_signal_bc)

    # Output the norm and slope signal
    output_fname = os.path.join(args.output_location, "{}.txt".format(args.output_prefix))
    f = open(output_fname, "w")
    f.write("\t".join((list(map(str, mean_signal_bias_f)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bias_r)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_raw)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_raw_f)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_raw_r)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bc)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bc_f)))) + "\n")
    f.write("\t".join((list(map(str, mean_signal_bc_r)))) + "\n")
    f.close()

    # Output PWM and create logo
    pwm_fname = os.path.join(args.output_location, "{}.pwm".format(args.output_prefix))
    pwm_file = open(pwm_fname, "w")
    for e in ["A", "C", "G", "T"]:
        pwm_file.write(" ".join([str(int(f)) for f in pwm_dict[e]]) + "\n")
    pwm_file.close()

    logo_fname = os.path.join(args.output_location, "{}.logo.eps".format(args.output_prefix))
    pwm = motifs.read(open(pwm_fname), "pfm")

    pwm.weblogo(logo_fname, format="eps", stack_width="large", stacks_per_line=str(args.window_size),
                color_scheme="color_classic", unit_name="", show_errorbars=False, logo_title="",
                show_xaxis=False, xaxis_label="", show_yaxis=False, yaxis_label="",
                show_fineprint=False, show_ends=False)

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(8, 8))
    fig, (ax1, ax4) = plt.subplots(2, figsize=(8, 4))
    start = -(args.window_size / 2)
    end = (args.window_size / 2) - 1
    x = np.linspace(start, end, num=args.window_size)

    ############################################################
    # bias signal per strand
    min_ = min(min(mean_signal_bias_f), min(mean_signal_bias_r))
    max_ = max(max(mean_signal_bias_f), max(mean_signal_bias_r))
    ax1.plot(x, mean_signal_bias_f, color='purple', label='Forward')
    ax1.plot(x, mean_signal_bias_r, color='green', label='Reverse')
    ax1.text(0.15, 0.9, 'n = {}'.format(num_sites), verticalalignment='bottom',
             horizontalalignment='right', transform=ax1.transAxes, fontweight='bold')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_position(('outward', 15))
    ax1.spines['bottom'].set_position(('outward', 5))
    ax1.tick_params(direction='out')
    ax1.set_xticks([start, 0, end])
    ax1.set_xticklabels([str(start), 0, str(end)])
    ax1.set_yticks([min_, max_])
    ax1.set_yticklabels([str(round(min_, 2)), str(round(max_, 2))], rotation=90)
    ax1.set_title(args.output_prefix, fontweight='bold')
    ax1.set_xlim(start, end)
    ax1.set_ylim([min_, max_])
    ax1.legend(loc="upper right", frameon=False)
    ####################################################################

    #####################################################################
    # Bias corrected, non-bias corrected (not strand specific)
    # min_ = min(min(mean_signal_raw_f), min(mean_signal_raw_r))
    # max_ = max(max(mean_signal_raw_f), max(mean_signal_raw_r))
    # ax2.plot(x, mean_signal_raw_f, color='red', label='Forward')
    # ax2.plot(x, mean_signal_raw_r, color='green', label='Reverse')
    # ax2.xaxis.set_ticks_position('bottom')
    # ax2.yaxis.set_ticks_position('left')
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['left'].set_position(('outward', 15))
    # ax2.tick_params(direction='out')
    # ax2.set_xticks([start, -1, 0, 1, end])
    # ax2.set_xticklabels([str(start), -1, 0,1, str(end)])
    # ax2.set_yticks([min_, max_])
    # ax2.set_yticklabels([str(round(min_, 2)), str(round(max_, 2))], rotation=90)
    # ax2.set_xlim(start, end)
    # ax2.set_ylim([min_, max_])
    # ax2.legend(loc="upper right", frameon=False)

    #####################################################################
    # Bias corrected and strand specific
    # min_ = min(min(mean_signal_bc_f), min(mean_signal_bc_r))
    # max_ = max(max(mean_signal_bc_f), max(mean_signal_bc_r))
    # ax3.plot(x, mean_signal_bc_f, color='red', label='Forward')
    # ax3.plot(x, mean_signal_bc_r, color='green', label='Reverse')
    # ax3.xaxis.set_ticks_position('bottom')
    # ax3.yaxis.set_ticks_position('left')
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['left'].set_position(('outward', 15))
    # ax3.tick_params(direction='out')
    # ax3.set_xticks([start, 0, end])
    # ax3.set_xticklabels([str(start), 0, str(end)])
    # ax3.set_yticks([min_, max_])
    # ax3.set_yticklabels([str(round(min_, 2)), str(round(max_, 2))], rotation=90)
    # ax3.set_xlim(start, end)
    # ax3.set_ylim([min_, max_])
    # ax3.legend(loc="upper right", frameon=False)

    #####################################################################
    # Bias corrected, non-bias corrected (not strand specific)
    min_ = min(min(mean_signal_raw), min(mean_signal_bc))
    max_ = max(max(mean_signal_raw), max(mean_signal_bc))
    ax4.plot(x, mean_signal_raw, color='blue', label='Uncorrected')
    ax4.plot(x, mean_signal_bc, color='red', label='Corrected')
    ax4.xaxis.set_ticks_position('bottom')
    ax4.yaxis.set_ticks_position('left')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_position(('outward', 15))
    ax4.tick_params(direction='out')
    ax4.set_xticks([start, 0, end])
    ax4.set_xticklabels([str(start), 0, str(end)])
    ax4.set_yticks([min_, max_])
    ax4.set_yticklabels([str(round(min_, 2)), str(round(max_, 2))], rotation=90)
    ax4.set_xlim(start, end)
    ax4.set_ylim([min_, max_])
    ax4.legend(loc="upper right", frameon=False)

    ax4.spines['bottom'].set_position(('outward', 40))

    ###############################################################################
    # merge the above figures
    figure_name = os.path.join(args.output_location, "{}.line.eps".format(args.output_prefix))
    fig.subplots_adjust(bottom=.2, hspace=.5)
    fig.tight_layout()
    fig.savefig(figure_name, format="eps", dpi=300)

    # Creating canvas and printing eps / pdf with merged results
    output_fname = os.path.join(args.output_location, "{}.eps".format(args.output_prefix))
    c = pyx.canvas.canvas()
    c.insert(pyx.epsfile.epsfile(0, 0, figure_name, scale=1.0))
    c.insert(pyx.epsfile.epsfile(1.45, 0.89, logo_fname, width=18.3, height=1.75))
    c.writeEPSfile(output_fname)
    os.system("epstopdf " + figure_name)
    os.system("epstopdf " + logo_fname)
    os.system("epstopdf " + output_fname)

    os.remove(pwm_fname)
    os.remove(os.path.join(args.output_location, "{}.line.eps".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.logo.eps".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.line.pdf".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.logo.pdf".format(args.output_prefix)))
    os.remove(os.path.join(args.output_location, "{}.eps".format(args.output_prefix)))


def fragment_size_raw_line(args):
    mpbs_regions = GenomicRegionSet("Motif Predicted Binding Sites")
    mpbs_regions.read(args.motif_file)
    bam = Samfile(args.reads_file, "rb")

    signal_f_max_145 = np.zeros(args.window_size)
    signal_r_max_145 = np.zeros(args.window_size)

    signal_f_146_307 = np.zeros(args.window_size)
    signal_r_146_307 = np.zeros(args.window_size)

    signal_f_min_307 = np.zeros(args.window_size)
    signal_r_min_307 = np.zeros(args.window_size)

    signal_f = np.zeros(args.window_size)
    signal_r = np.zeros(args.window_size)

    for region in mpbs_regions:
        if str(region.name).split(":")[-1] == "Y":
            # Extend by 50 bp
            mid = (region.initial + region.final) / 2
            p1 = mid - (args.window_size / 2)
            p2 = mid + (args.window_size / 2)

            # Fetch raw signal
            for read in bam.fetch(region.chrom, p1, p2):
                # All reads
                if not read.is_reverse:
                    cut_site = read.pos + args.forward_shift
                    if p1 <= cut_site < p2:
                        signal_f[cut_site - p1] += 1.0
                else:
                    cut_site = read.aend + args.reverse_shift - 1
                    if p1 <= cut_site < p2:
                        signal_r[cut_site - p1] += 1.0

                # length <= 145
                if abs(read.template_length) <= 145:
                    if not read.is_reverse:
                        cut_site = read.pos + args.forward_shift
                        if p1 <= cut_site < p2:
                            signal_f_max_145[cut_site - p1] += 1.0
                    else:
                        cut_site = read.aend + args.reverse_shift - 1
                        if p1 <= cut_site < p2:
                            signal_r_max_145[cut_site - p1] += 1.0

                # length > 145 and <= 307
                if 145 < abs(read.template_length) <= 307:
                    if not read.is_reverse:
                        cut_site = read.pos + args.forward_shift
                        if p1 <= cut_site < p2:
                            signal_f_146_307[cut_site - p1] += 1.0
                    else:
                        cut_site = read.aend + args.reverse_shift - 1
                        if p1 <= cut_site < p2:
                            signal_r_146_307[cut_site - p1] += 1.0

                # length > 307
                if abs(read.template_length) > 307:
                    if not read.is_reverse:
                        cut_site = read.pos + args.forward_shift
                        if p1 <= cut_site < p2:
                            signal_f_min_307[cut_site - p1] += 1.0
                    else:
                        cut_site = read.aend + args.reverse_shift - 1
                        if p1 <= cut_site < p2:
                            signal_r_min_307[cut_site - p1] += 1.0

    # Output the norm and slope signal
    output_fname = os.path.join(args.output_location, "{}.txt".format(args.output_prefix))
    f = open(output_fname, "w")
    f.write("\t".join((list(map(str, signal_f)))) + "\n")
    f.write("\t".join((list(map(str, signal_r)))) + "\n")
    f.write("\t".join((list(map(str, signal_f_max_145)))) + "\n")
    f.write("\t".join((list(map(str, signal_r_max_145)))) + "\n")
    f.write("\t".join((list(map(str, signal_f_146_307)))) + "\n")
    f.write("\t".join((list(map(str, signal_r_146_307)))) + "\n")
    f.write("\t".join((list(map(str, signal_f_min_307)))) + "\n")
    f.write("\t".join((list(map(str, signal_r_min_307)))) + "\n")
    f.close()

    # find out the linker position
    pos_f_1, pos_r_1, pos_f_2, pos_r_2 = get_linkers_position(signal_f_146_307,
                                                              signal_r_146_307,
                                                              signal_f_min_307,
                                                              signal_r_min_307)
    p1 = (pos_f_1 - pos_f_2) / 2 + pos_f_2
    p2 = p1 + 180
    p3 = args.window_size - p2
    p4 = args.window_size - p1

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))

    start = -(args.window_size / 2)
    end = (args.window_size / 2) - 1
    x = np.linspace(start, end, num=args.window_size)
    x_ticks = [start, p1 - 500, p2 - 500, 0, p3 - 500, p4 - 500, end]

    update_axes_for_fragment_size_line(ax1, x, x_ticks, start, end, signal_f, signal_r, p1, p2,
                                       p3, p4)
    update_axes_for_fragment_size_line(ax2, x, x_ticks, start, end, signal_f_max_145, signal_r_max_145, p1, p2,
                                       p3, p4)
    update_axes_for_fragment_size_line(ax3, x, x_ticks, start, end, signal_f_146_307, signal_r_146_307, p1, p2,
                                       p3, p4)
    update_axes_for_fragment_size_line(ax4, x, x_ticks, start, end, signal_f_min_307, signal_r_min_307, p1, p2,
                                       p3, p4)

    figure_name = os.path.join(args.output_location, "{}.pdf".format(args.output_prefix))
    fig.subplots_adjust(bottom=.2, hspace=.5)
    fig.tight_layout()
    fig.savefig(figure_name, format="pdf", dpi=300)


def fragment_size_bc_line(args):
    mpbs_regions = GenomicRegionSet("Motif Predicted Binding Sites")
    mpbs_regions.read(args.motif_file)

    genomic_signal = GenomicSignal(args.reads_file)
    genomic_signal.load_sg_coefs(11)
    bam = Samfile(args.reads_file, "rb")
    genome_data = GenomeData(args.organism)
    fasta = Fastafile(genome_data.get_genome())

    bias_table = BiasTable()
    bias_table_list = args.bias_table.split(",")
    table = bias_table.load_table(table_file_name_F=bias_table_list[0],
                                  table_file_name_R=bias_table_list[1])

    signal_f_max_145 = np.zeros(args.window_size)
    signal_r_max_145 = np.zeros(args.window_size)

    signal_f_146_307 = np.zeros(args.window_size)
    signal_r_146_307 = np.zeros(args.window_size)

    signal_f_min_307 = np.zeros(args.window_size)
    signal_r_min_307 = np.zeros(args.window_size)

    signal_f = np.zeros(args.window_size)
    signal_r = np.zeros(args.window_size)

    for region in mpbs_regions:
        if str(region.name).split(":")[-1] == "Y":
            mid = (region.initial + region.final) / 2
            p1 = mid - (args.window_size / 2)
            p2 = mid + (args.window_size / 2)

            # All reads
            signal_bc_f, signal_bc_r = \
                genomic_signal.get_bc_signal_by_fragment_length(ref=region.chrom, start=p1, end=p2,
                                                                bam=bam, fasta=fasta,
                                                                bias_table=table,
                                                                forward_shift=args.forward_shift,
                                                                reverse_shift=args.reverse_shift,
                                                                min_length=None, max_length=None,
                                                                strand=True)
            # length <= 145
            signal_bc_max_145_f, signal_bc_max_145_r = \
                genomic_signal.get_bc_signal_by_fragment_length(ref=region.chrom, start=p1, end=p2,
                                                                bam=bam, fasta=fasta,
                                                                bias_table=table,
                                                                forward_shift=args.forward_shift,
                                                                reverse_shift=args.reverse_shift,
                                                                min_length=None, max_length=145,
                                                                strand=True)
            # length > 145 and <= 307
            signal_bc_146_307_f, signal_bc_146_307_r = \
                genomic_signal.get_bc_signal_by_fragment_length(ref=region.chrom, start=p1, end=p2,
                                                                bam=bam, fasta=fasta,
                                                                bias_table=table,
                                                                forward_shift=args.forward_shift,
                                                                reverse_shift=args.reverse_shift,
                                                                min_length=145, max_length=307,
                                                                strand=True)
            # length > 307
            signal_bc_min_307_f, signal_bc_min_307_r = \
                genomic_signal.get_bc_signal_by_fragment_length(ref=region.chrom, start=p1, end=p2,
                                                                bam=bam, fasta=fasta,
                                                                bias_table=table,
                                                                forward_shift=args.forward_shift,
                                                                reverse_shift=args.reverse_shift,
                                                                min_length=307, max_length=None,
                                                                strand=True)

            signal_f = np.add(signal_f, np.array(signal_bc_f))
            signal_r = np.add(signal_r, np.array(signal_bc_r))
            signal_f_max_145 = np.add(signal_f_max_145, np.array(signal_bc_max_145_f))
            signal_r_max_145 = np.add(signal_r_max_145, np.array(signal_bc_max_145_r))
            signal_f_146_307 = np.add(signal_f_146_307, np.array(signal_bc_146_307_f))
            signal_r_146_307 = np.add(signal_r_146_307, np.array(signal_bc_146_307_r))
            signal_f_min_307 = np.add(signal_f_min_307, np.array(signal_bc_min_307_f))
            signal_r_min_307 = np.add(signal_r_min_307, np.array(signal_bc_min_307_r))

    # Output the norm and slope signal
    output_fname = os.path.join(args.output_location, "{}.txt".format(args.output_prefix))
    f = open(output_fname, "w")
    f.write("\t".join((list(map(str, signal_f)))) + "\n")
    f.write("\t".join((list(map(str, signal_r)))) + "\n")
    f.write("\t".join((list(map(str, signal_f_max_145)))) + "\n")
    f.write("\t".join((list(map(str, signal_r_max_145)))) + "\n")
    f.write("\t".join((list(map(str, signal_f_146_307)))) + "\n")
    f.write("\t".join((list(map(str, signal_r_146_307)))) + "\n")
    f.write("\t".join((list(map(str, signal_f_min_307)))) + "\n")
    f.write("\t".join((list(map(str, signal_r_min_307)))) + "\n")
    f.close()

    # find out the linker position
    pos_f_1, pos_r_1, pos_f_2, pos_r_2 = get_linkers_position(signal_f_146_307,
                                                              signal_r_146_307,
                                                              signal_f_min_307,
                                                              signal_r_min_307)
    p1 = (pos_f_1 - pos_f_2) / 2 + pos_f_2
    p2 = p1 + 180
    p3 = args.window_size - p2
    p4 = args.window_size - p1

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))

    start = -(args.window_size / 2)
    end = (args.window_size / 2) - 1
    x = np.linspace(start, end, num=args.window_size)
    x_ticks = [start, p1 - 500, p2 - 500, 0, p3 - 500, p4 - 500, end]

    update_axes_for_fragment_size_line(ax1, x, x_ticks, start, end, signal_f, signal_r, p1, p2, p3, p4)
    update_axes_for_fragment_size_line(ax2, x, x_ticks, start, end, signal_f_max_145, signal_r_max_145, p1, p2,
                                       p3, p4)
    update_axes_for_fragment_size_line(ax3, x, x_ticks, start, end, signal_f_146_307, signal_r_146_307, p1, p2,
                                       p3, p4)
    update_axes_for_fragment_size_line(ax4, x, x_ticks, start, end, signal_f_min_307, signal_r_min_307, p1, p2,
                                       p3, p4)

    figure_name = os.path.join(args.output_location, "{}.pdf".format(args.output_prefix))
    fig.subplots_adjust(bottom=.2, hspace=.5)
    fig.tight_layout()
    fig.savefig(figure_name, format="pdf", dpi=300)


def update_axes_for_fragment_size_line(ax, x, x_ticks, start, end, signal_f, signal_r, p1, p2, p3, p4):
    max_signal = max(max(signal_f), max(signal_r))
    min_signal = min(min(signal_f), min(signal_r))
    ax.plot(x, signal_f, color='red', label='Forward')
    ax.plot(x, signal_r, color='green', label='Reverse')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 15))
    ax.tick_params(direction='out')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(list(map(str, x_ticks)))
    ax.set_xlim(start, end)
    ax.set_yticks([min_signal, max_signal])
    ax.set_yticklabels([str(int(min_signal)), str(int(max_signal))], rotation=90)
    ax.set_ylim([min_signal, max_signal])
    ax.legend().set_visible(False)
    f_1, r_1 = sum(signal_f[:p1]) / sum(signal_r), sum(signal_r[:p1]) / sum(signal_r)
    f_2, r_2 = sum(signal_f[p1:p2]) / sum(signal_r), sum(signal_r[p1:p2]) / sum(signal_r)
    f_3, r_3 = sum(signal_f[p2:500]) / sum(signal_r), sum(signal_r[p2:500]) / sum(signal_r)
    f_4, r_4 = sum(signal_f[500:p3]) / sum(signal_r), sum(signal_r[500:p3]) / sum(signal_r)
    f_5, r_5 = sum(signal_f[p3:p4]) / sum(signal_r), sum(signal_r[p3:p4]) / sum(signal_r)
    f_6, r_6 = sum(signal_f[p4:]) / sum(signal_r), sum(signal_r[p4:]) / sum(signal_r)
    text_x_1 = ((p1 - 0) / 2.0 + 0) / 1000
    text_x_2 = ((p2 - p1) / 2.0 + p1) / 1000
    text_x_3 = ((500 - p2) / 2.0 + p2) / 1000
    text_x_4 = ((p3 - 500) / 2.0 + 500) / 1000
    text_x_5 = ((p4 - p3) / 2.0 + p3) / 1000
    text_x_6 = ((1000 - p4) / 2.0 + p4) / 1000
    ax.text(text_x_1, 1.0, str(round(f_1, 2)), verticalalignment='center', color='red',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_1, 0.9, str(round(r_1, 2)), verticalalignment='center', color='green',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_2, 1.0, str(round(f_2, 2)), verticalalignment='center', color='red',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_2, 0.9, str(round(r_2, 2)), verticalalignment='center', color='green',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_3, 1.0, str(round(f_3, 2)), verticalalignment='center', color='red',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_3, 0.9, str(round(r_3, 2)), verticalalignment='center', color='green',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_4, 1.0, str(round(f_4, 2)), verticalalignment='center', color='red',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_4, 0.9, str(round(r_4, 2)), verticalalignment='center', color='green',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_5, 1.0, str(round(f_5, 2)), verticalalignment='center', color='red',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_5, 0.9, str(round(r_5, 2)), verticalalignment='center', color='green',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_6, 1.0, str(round(f_6, 2)), verticalalignment='center', color='red',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.text(text_x_6, 0.9, str(round(r_6, 2)), verticalalignment='center', color='green',
            horizontalalignment='center', transform=ax.transAxes, fontsize=12)


def get_linkers_position(signal_f_146_307, signal_r_146_307, signal_f_min_307, signal_r_min_307):
    smooth_signal_f_146_307 = savgol_filter(signal_f_146_307, window_length=51, polyorder=2)
    smooth_signal_r_146_307 = savgol_filter(signal_r_146_307, window_length=51, polyorder=2)
    smooth_signal_f_min_307 = savgol_filter(signal_f_min_307, window_length=51, polyorder=2)
    smooth_signal_r_min_307 = savgol_filter(signal_r_min_307, window_length=51, polyorder=2)

    position_f_1 = np.argmax(smooth_signal_f_146_307[:400])
    position_f_2 = np.argmax(smooth_signal_f_min_307[:position_f_1])

    position_r_1 = np.argmax(smooth_signal_r_146_307[600:]) + 600
    position_r_2 = np.argmax(smooth_signal_r_min_307[position_r_1:]) + position_r_1

    return position_f_1, position_r_1, position_f_2, position_r_2


def rescaling(vector):
    maxN = max(vector)
    minN = min(vector)
    return [(e - minN) / (maxN - minN) for e in vector]
