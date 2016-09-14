#! /bin/zsh

setopt shwordsplit

src="/homes/jsanterre/data/raw/amr/Mycobacterium/kanamycin/"
tgt="/homes/jsanterre/data/raw/amr/Mycobacterium/kanamycin/counts"

kmers="10 "
dirs="All"
tmpd="/homes/jsanterre/data/raw/amr/Mycobacterium/kanamycin/tmp"

for k in $kmers; do
    for d in $dirs; do
        dd="$tgt/$d/k$k"
        echo $dd $k
        mkdir -p $dd
        pushd $dd; ls $src/$d/* | parallel -j 1 "kc.pl -fm -fa -tmp $tmpd -k $k {}"; popd
    done
done