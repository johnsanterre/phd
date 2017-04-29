#! /bin/zsh

setopt shwordsplit
src="/homes/jsanterre/data/raw/amr/Mycobacterium/isoniazid"
tgt="/homes/jsanterre/data/raw/amr/Mycobacterium/isoniazid/counts"

kmers="17 18 19 20"
dirs="All"
tmpd="/homes/jsanterre/data/raw/amr/Mycobacterium/isoniazid/tmp"

for k in $kmers; do
    for d in $dirs; do
        dd="$tgt/$d/k$k"
        echo $dd $k
        mkdir -p $dd
        pushd $dd; ls $src/$d/* | parallel -j 1 "perl /homes/jsanterre/phd/anl_scripts/kc.pl -fm -tmp $tmpd -k $k {}"; popd
    done
done
