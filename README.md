fastq2parts.py -i in.fastq -o workdir/parts -p 2000000
bowtie_parts_with_LSF.py -i workdir/parts -o workdir/alns -x path/to/index_name.ebwt -l workdir/logs_aln -q short_serial
parts2barcodes.py -i workdir/alns -o workdir/barcodes -m mapping.txt

alns2counts.py -i workdir/barcodes -o workdir/counts.csv -r input_counts.csv
# OR
alns2counts_separated.py -i workdir/barcodes -o workdir/counts -r input_counts.csv

counts2pvals.py -i workdir/counts.csv -o workdir/pvals.csv
# OR
counts2pvals_separated.py -i workdir/counts -o workdir/pvals -q short_serial -l logs_pvals