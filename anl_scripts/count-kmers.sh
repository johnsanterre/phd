#! /bin/zsh

setopt shwordsplit

src="/disks/jul302015vol/fangfang/amr/maela.sub100"
tgt="/disks/jul302015vol/fangfang/amr/maela.sub100/counts"

#kmers="10 15 20 25 30"
kmers="11 12 13 14"
dirs="genomes"
tmpd="/tmp"

for k in $kmers; do
    for d in $dirs; do
        dd="$tgt/$d/k$k"
        echo $dd $k
        mkdir -p $dd
        pushd $dd; ls $src/$d/* | parallel -j 1 "kc.pl -fm -fa -tmp $tmpd -k $k {}"; popd
    done
done