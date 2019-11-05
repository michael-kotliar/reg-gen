import os
from argparse import SUPPRESS
import numpy as np
import pysam
import time

# Internal
from rgt.Util import GenomeData, ErrorHandler
from rgt.GenomicRegionSet import GenomicRegionSet
from rgt.HINT.GenomicSignal import GenomicSignal


def tracks_args(parser):
    # Parameters Options
    parser.add_argument("--organism", type=str, metavar="STRING", default="hg19",
                        help="Organism considered on the analysis. Must have been setup in the RGTDATA folder. "
                             "Common choices are hg19, hg38. mm9, and mm10. DEFAULT: hg19")
    parser.add_argument("--bias-table", type=str, metavar="FILE1_F,FILE1_R", default=None,
                        help="Bias table files used to generate bias corrected tracks. DEFAULT: None")

    # Hidden Options
    parser.add_argument("--initial-clip", type=int, metavar="INT", default=1000, help=SUPPRESS)
    parser.add_argument("--forward-shift", type=int, metavar="INT", default=4, help=SUPPRESS)
    parser.add_argument("--reverse-shift", type=int, metavar="INT", default=-5, help=SUPPRESS)
    parser.add_argument("--norm-window", type=int, metavar="INT", default=100, help=SUPPRESS)
    parser.add_argument("--k-nb", type=int, metavar="INT", default=6, help=SUPPRESS)

    # Output Options
    parser.add_argument("--raw", action="store_true", default=False,
                        help="If set, the raw signals from DNase-seq or ATAC-seq data will be generated. "
                             "DEFAULT: False")
    parser.add_argument("--raw-norm", action="store_true", default=False,
                        help="If set, the normalized raw signals from DNase-seq or ATAC-seq data will be generated. "
                             "DEFAULT: False")
    parser.add_argument("--bc", action="store_true", default=False,
                        help="If set, the bias corrected signals from DNase-seq or ATAC-seq data will be generated. "
                             "DEFAULT: False")
    parser.add_argument("--norm", action="store_true", default=False,
                        help="If set, the normalised signals from DNase-seq or ATAC-seq data will be generated. "
                             "DEFAULT: False")
    parser.add_argument("--bigWig", action="store_true", default=False,
                        help="If set, all .wig files will be converted to .bw files. DEFAULT: False")
    parser.add_argument("--strand-specific", action="store_true", default=False,
                        help="If set, the tracks will be splitted into two files, one for forward and another for "
                             "reverse strand. DEFAULT: False")

    # Output Options
    parser.add_argument("--output-location", type=str, metavar="PATH", default=os.getcwd(),
                        help="Path where the output bias table files will be written. DEFAULT: current directory")
    parser.add_argument("--output-prefix", type=str, metavar="FILENAME", default="tracks",
                        help="The prefix for results files. DEFAULT: tracks")

    parser.add_argument('input_files', metavar='reads.bam regions.bed', type=str, nargs='*',
                        help='BAM file of reads and BED files of interesting regions')


def tracks_run(args):
    if args.raw:
        get_raw_tracks(args=args)
    elif args.raw_norm:
        get_raw_norm_tracks(args=args)


def get_raw_tracks(args):
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
    regions.merge()
    genomic_signal = GenomicSignal(args.input_files[0])

    print("{}: generating signal for {} regions...\n".format(time.strftime("%D-%H:%M:%S"),
                                                             len(regions)))
    if args.strand_specific:
        output_forward_filename = os.path.join(args.output_location, "{}_forward.wig".format(args.output_prefix))
        output_reverse_filename = os.path.join(args.output_location, "{}_reverse.wig".format(args.output_prefix))
        output_forward_f = open(output_forward_filename, "a")
        output_reverse_f = open(output_reverse_filename, "a")

        for region in regions:
            signal_forward, signal_reverse = genomic_signal.get_raw_signal(chromosome=region.chrom,
                                                                           start=region.initial,
                                                                           end=region.final,
                                                                           forward_shift=args.forward_shift,
                                                                           reverse_shift=args.reverse_shift,
                                                                           initial_clip=args.initial_clip,
                                                                           strand_specific=True)

            output_forward_f.write(
                "fixedStep chrom=" + region.chrom + " start=" + str(region.initial + 1) + " step=1\n" +
                "\n".join([str(e) for e in np.nan_to_num(signal_forward)]) + "\n")
            output_reverse_f.write(
                "fixedStep chrom=" + region.chrom + " start=" + str(region.initial + 1) + " step=1\n" +
                "\n".join([str(e) for e in np.nan_to_num(signal_reverse)]) + "\n")

        output_forward_f.close()
        output_reverse_f.close()

        if args.bigWig:
            genome_data = GenomeData(args.organism)
            chrom_sizes_file = genome_data.get_chromosome_sizes()
            bw_forward_filename = os.path.join(args.output_location, "{}_forward.bw".format(args.output_prefix))
            bw_reverse_filename = os.path.join(args.output_location, "{}_reverse.bw".format(args.output_prefix))
            os.system(
                " ".join(["wigToBigWig", output_forward_filename, chrom_sizes_file, bw_forward_filename, "-verbose=0"]))
            os.system(
                " ".join(["wigToBigWig", output_reverse_filename, chrom_sizes_file, bw_reverse_filename, "-verbose=0"]))
            os.remove(output_forward_filename)
            os.remove(output_reverse_filename)
    else:
        output_filename = os.path.join(args.output_location, "{}.wig".format(args.output_prefix))
        output_f = open(output_filename, "a")

        for region in regions:
            signal = genomic_signal.get_raw_signal(chromosome=region.chrom,
                                                   start=region.initial,
                                                   end=region.final,
                                                   forward_shift=args.forward_shift,
                                                   reverse_shift=args.reverse_shift,
                                                   initial_clip=args.initial_clip,
                                                   strand_specific=False)

            output_f.write("fixedStep chrom=" + region.chrom + " start=" + str(region.initial + 1) + " step=1\n" +
                           "\n".join([str(e) for e in np.nan_to_num(signal)]) + "\n")
        output_f.close()

        if args.bigWig:
            genome_data = GenomeData(args.organism)
            chrom_sizes_file = genome_data.get_chromosome_sizes()
            bw_filename = os.path.join(args.output_location, "{}.bw".format(args.output_prefix))
            os.system(" ".join(["wigToBigWig", output_filename, chrom_sizes_file, bw_filename, "-verbose=0"]))
            os.remove(output_filename)


def get_raw_norm_tracks(args):
    # Initializing Error Handler
    err = ErrorHandler()

    if len(args.input_files) != 2:
        err.throw_error("ME_FEW_ARG", add_msg="You must specify reads and regions file.")

    output_fname = os.path.join(args.output_location, "{}.wig".format(args.output_prefix))

    bam_file, region_file = args.input_files[0], args.input_files[1]

    # check if index exists for bam file
    bam_index_file = "{}.bai".format(bam_file)
    if not os.path.exists(bam_index_file):
        pysam.index(bam_file)

    regions = GenomicRegionSet("Interested regions")
    regions.read(region_file)
    regions.merge()
    genomic_signal = GenomicSignal(args.input_files[0])

    print("{}: generating signal for {} regions...\n".format(time.strftime("%D-%H:%M:%S"),
                                                             len(regions)))
    with open(output_fname, "a") as output_f:
        for region in regions:
            signal = genomic_signal.get_raw_norm_signal(chromosome=region.chrom,
                                                        start=region.initial,
                                                        end=region.final,
                                                        forward_shift=args.forward_shift,
                                                        reverse_shift=args.reverse_shift,
                                                        initial_clip=args.initial_clip,
                                                        norm_window=args.norm_window)

            output_f.write("fixedStep chrom=" + region.chrom + " start=" + str(region.initial + 1) + " step=1\n" +
                           "\n".join([str(e) for e in np.nan_to_num(signal)]) + "\n")
    output_f.close()

    if args.bigWig:
        genome_data = GenomeData(args.organism)
        chrom_sizes_file = genome_data.get_chromosome_sizes()
        bw_filename = os.path.join(args.output_location, "{}.bw".format(args.output_prefix))
        os.system(" ".join(["wigToBigWig", output_fname, chrom_sizes_file, bw_filename, "-verbose=0"]))
        os.remove(output_fname)

# def get_bc_tracks(args):
#     # Initializing Error Handler
#     err = ErrorHandler()
#
#     if len(args.input_files) != 2:
#         err.throw_error("ME_FEW_ARG", add_msg="You must specify reads and regions file.")
#
#     regions = GenomicRegionSet("Interested regions")
#     regions.read(args.input_files[1])
#     regions.merge()
#
#     reads_file = GenomicSignal()
#
#     bam = Samfile(args.input_files[0], "rb")
#     genome_data = GenomeData(args.organism)
#     fasta = Fastafile(genome_data.get_genome())
#
#     hmm_data = HmmData()
#     if args.bias_table:
#         bias_table_list = args.bias_table.split(",")
#         bias_table = BiasTable().load_table(table_file_name_F=bias_table_list[0],
#                                             table_file_name_R=bias_table_list[1])
#     else:
#         table_F = hmm_data.get_default_bias_table_F_ATAC()
#         table_R = hmm_data.get_default_bias_table_R_ATAC()
#         bias_table = BiasTable().load_table(table_file_name_F=table_F,
#                                             table_file_name_R=table_R)
#
#     if args.strand_specific:
#         fname_forward = os.path.join(args.output_location, "{}_forward.wig".format(args.output_prefix))
#         fname_reverse = os.path.join(args.output_location, "{}_reverse.wig".format(args.output_prefix))
#
#         f_forward = open(fname_forward, "a")
#         f_reverse = open(fname_reverse, "a")
#         for region in regions:
#             signal_f, signal_r = reads_file.get_bc_signal_by_fragment_length(
#                 ref=region.chrom, start=region.initial, end=region.final, bam=bam, fasta=fasta, bias_table=bias_table,
#                 forward_shift=args.forward_shift, reverse_shift=args.reverse_shift, min_length=None, max_length=None,
#                 strand=True)
#
#             if args.norm:
#                 signal_f = reads_file.boyle_norm(signal_f)
#                 perc = scoreatpercentile(signal_f, 98)
#                 std = np.std(signal_f)
#                 signal_f = reads_file.hon_norm_atac(signal_f, perc, std)
#
#                 signal_r = reads_file.boyle_norm(signal_r)
#                 perc = scoreatpercentile(signal_r, 98)
#                 std = np.std(signal_r)
#                 signal_r = reads_file.hon_norm_atac(signal_r, perc, std)
#
#             f_forward.write("fixedStep chrom=" + region.chrom + " start=" + str(region.initial + 1) + " step=1\n" +
#                             "\n".join([str(e) for e in np.nan_to_num(signal_f)]) + "\n")
#
#             f_reverse.write("fixedStep chrom=" + region.chrom + " start=" + str(region.initial + 1) + " step=1\n" +
#                             "\n".join([str(-e) for e in np.nan_to_num(signal_r)]) + "\n")
#
#         f_forward.close()
#         f_reverse.close()
#
#         if args.bigWig:
#             genome_data = GenomeData(args.organism)
#             chrom_sizes_file = genome_data.get_chromosome_sizes()
#
#             bw_filename = os.path.join(args.output_location, "{}_forward.bw".format(args.output_prefix))
#             os.system(" ".join(["wigToBigWig", fname_forward, chrom_sizes_file, bw_filename, "-verbose=0"]))
#             os.remove(fname_forward)
#
#             bw_filename = os.path.join(args.output_location, "{}_reverse.bw".format(args.output_prefix))
#             os.system(" ".join(["wigToBigWig", fname_reverse, chrom_sizes_file, bw_filename, "-verbose=0"]))
#             os.remove(fname_reverse)
#
#     else:
#         output_fname = os.path.join(args.output_location, "{}.wig".format(args.output_prefix))
#         with open(output_fname, "a") as output_f:
#             for region in regions:
#                 signal = reads_file.get_bc_signal_by_fragment_length(ref=region.chrom, start=region.initial,
#                                                                      end=region.final,
#                                                                      bam=bam, fasta=fasta, bias_table=bias_table,
#                                                                      forward_shift=args.forward_shift,
#                                                                      reverse_shift=args.reverse_shift,
#                                                                      min_length=None, max_length=None, strand=False)
#
#                 if args.norm:
#                     signal = reads_file.boyle_norm(signal)
#                     perc = scoreatpercentile(signal, 98)
#                     std = np.std(signal)
#                     signal = reads_file.hon_norm_atac(signal, perc, std)
#
#                 output_f.write("fixedStep chrom=" + region.chrom + " start=" + str(region.initial + 1) + " step=1\n" +
#                                "\n".join([str(e) for e in np.nan_to_num(signal)]) + "\n")
#         output_f.close()
#
#         if args.bigWig:
#             genome_data = GenomeData(args.organism)
#             chrom_sizes_file = genome_data.get_chromosome_sizes()
#             bw_filename = os.path.join(args.output_location, "{}.bw".format(args.output_prefix))
#             os.system(" ".join(["wigToBigWig", output_fname, chrom_sizes_file, bw_filename, "-verbose=0"]))
#             os.remove(output_fname)
