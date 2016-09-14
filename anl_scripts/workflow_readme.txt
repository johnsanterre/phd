Work Flow

1. Connect to larch(kmerge requires larch environment).

2. Create single directory with all .fna files in one directory, while making a  bateriaName_antibioticName_isolates.log file with this format.
KPN124ec        S
KPN125ec        R

3. kc.pl currently lives at /homes/jsanterre/phd/anl_scripts/kc.pl

4. To run a single run of kc.pl do this
perl /homes/jsanterre/phd/anl_scripts/kc.pl -fm -tmp ./tmp -k 10 mycobacterium_kanamycin_isolates.log




cur dir = /homes/jsanterre/data/raw/amr/Mycobacterium/kanamycin