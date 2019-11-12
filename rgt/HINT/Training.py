from argparse import SUPPRESS
import time

# Internal
from rgt.GenomicRegionSet import GenomicRegionSet
from rgt.Util import ErrorHandler
from rgt.HINT.GenomicSignal import GenomicSignal
from rgt.HINT.HMM import HMM
from rgt.HINT.BiasTable import BiasTable
from rgt.HINT.Util import *

"""
Train a hidden Markov model (HMM) based on the annotation data

Authors: Eduardo G. Gusmao, Zhijian Li
"""


def training_args(parser):
    # Parameters Options
    parser.add_argument("--organism", type=str, metavar="STRING", default="hg19")
    parser.add_argument("--bias-table", type=str, metavar="FILE1_F,FILE1_R", default=None)
    parser.add_argument("--num-states", type=int, metavar="INT", default=7,
                        help="The states number of HMM model.")

    parser.add_argument("--raw", action="store_true", default=False,
                        help="Train a HMM with raw signals from DNase-seq or ATAC-seq data"
                             "DEFAULT: False")
    # Train Options
    parser.add_argument("--n-iter", type=int, metavar="INT", default=100,
                        help="Maximum number of iterations to perform.")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="When True, per-iteration convergence reports are printed.")

    # Hidden Options
    parser.add_argument("--forward-shift", type=int, metavar="INT", default=0, help=SUPPRESS)
    parser.add_argument("--reverse-shift", type=int, metavar="INT", default=0, help=SUPPRESS)
    parser.add_argument("--k-nb", type=int, metavar="INT", default=6, help=SUPPRESS)
    parser.add_argument("--extend-window", type=int, metavar="INT", default=100, help=SUPPRESS)

    # Output Options
    parser.add_argument("--output-location", type=str, metavar="PATH", default=os.getcwd(),
                        help="Path where the output bias table files will be written.")
    parser.add_argument("--output-prefix", type=str, metavar="STRING", default=None,
                        help="The prefix for results files.")

    parser.add_argument('input_files', metavar='reads.bam regions.bed', type=str, nargs='*',
                        help='BAM file of reads and BED files of binding sites for training')


def training_run(args):
    if args.raw:
        train_raw(args)


def get_raw_signal(args):
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

    chrom_sizes_dict = get_chromosome_size(args.organism)

    print("{}: generating signal for {} regions...\n".format(time.strftime("%D-%H:%M:%S"),
                                                             len(regions)))

    signals, slopes, lengths = list(), list(), list()
    for i, region in enumerate(regions):
        start = max(0, region.initial - args.extend_window)
        end = min(region.final + args.extend_window, chrom_sizes_dict[region.chrom])
        signal, slope = genomic_signal.get_raw_norm_signal(chromosome=region.chrom,
                                                           chromosome_size=chrom_sizes_dict[region.chrom],
                                                           start=start,
                                                           end=end,
                                                           forward_shift=args.forward_shift,
                                                           reverse_shift=args.reverse_shift)

        if np.isnan(signal).any() or np.isnan(slope).any():
            continue
        else:
            signals.extend(signal)
            slopes.extend(slope)
            lengths.append(end - start)

    return signals, slopes, lengths


def train_raw(args):
    signals, slopes, lengths = get_raw_signal(args=args)

    assert len(signals) == len(slopes), "The length of signal is {} while length slope is {}".format(len(signals),
                                                                                                     len(slopes))

    hmm = HMM(n_components=args.num_states, random_state=42, n_iter=args.n_iter,
              covariance_type="full", fp_state=args.num_states - 1,
              verbose=args.verbose, params="stmc", init_params="stmc")

    hmm.fit(X=np.array([signals, slopes], dtype=np.float32).T, lengths=lengths)

    output_file_name = os.path.join(args.output_location, "{}.hmm".format(args.output_prefix))
    hmm.save_hmm(output_file_name)

    hmm.load_hmm(output_file_name)

    # # make sure covariance is symmetric and positive-definite
    # for i in range(hmm_model.n_components):
    #     while np.any(np.array(linalg.eigvalsh(hmm_model.covars_[i])) <= 0):
    #         hmm_model.covars_[i] += 0.000001 * np.eye(hmm_model.covars_[i].shape[0])
    #
    # output_fname = os.path.join(args.output_location, "{}.pkl".format(args.output_prefix))
    # joblib.dump(hmm_model, output_fname)
