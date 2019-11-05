import pysam

# Internal
from rgt.Util import AuxiliaryFunctions, GenomeData


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
