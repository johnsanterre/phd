#! /usr/bin/env perl

use strict;
use Carp;
use Data::Dumper;
use File::Temp qw(tempdir);

use Getopt::Long;

my ($help, $show_anno, $show_neighbors, $ref_file, $gff_file);

my $usage = "Usage: $0 [-a] [-n] kmer_file\n\n";

GetOptions("h|help" => \$help,
           "a"      => \$show_anno,
           "n"      => \$show_neighbors,
           "ref=s"  => \$ref_file,
           "gff=s"  => \$gff_file
	  ) or die("Error in command line arguments\n");

$help and die $usage;

my $ref = $ref_file || '/bigdata/fangfang/amr/search/ref.fa';
my $gff = $gff_file || '/bigdata/fangfang/amr/search/ref.gff';
my $kmerF = shift @ARGV or die $usage;

my @kmers = grep { length >= 15 } split(/[^ATGC]+/, `cat $kmerF`);
my $id = 0;
my $query = join('', map { ">kmer".++$id."\n$_\n" } @kmers);

my $features;
if ($show_anno) {
    $features = get_sorted_features($ref, $gff);
}

my $tmpdir = tempdir(CLEANUP => 1);
my $queryF = "$tmpdir/query.fa";
open(FQ, ">$queryF") or die "Could not open $queryF";
print FQ $query;
close(FQ);

my $bwa = '/home/fangfang/bin/bwa';
-s $ref.".bwt" or run("$bwa index $ref");
open(SAM, "$bwa mem -T15 -t16 -k10 -B1 -O1 $ref $queryF 2>/dev/null |");
while (<SAM>) {
    next if /\@/;
    my @c = split /\t/;
    next if $c[1] & 0x4;
    my $query = $c[0];
    my $strand = $c[1] & 0x10 ? '-' : '+';
    my $subject = $c[2];
    my $start = $c[3];
    my $len = length($c[9]);
    my ($score) = $c[13] =~ /AS:i:(\d+)/;
    $score = sprintf("%.3f", $score / $len) if $score;
    my @new = ($query, $strand, $subject, $start, $len, $score);
    if ($show_anno) {
        my $info = feature_info_for_position($subject, $start, $features);
        push @new, $info->{gene}->[0];
        push @new, $info->{gene}->[8];
        if ($show_neighbors) {
            push @new, $info->{left}->[0];
            push @new, $info->{left}->[8];
            push @new, $info->{right}->[0];
            push @new, $info->{right}->[8];
        }
    }
    print join("\t", @new)."\n";
}
close(SAM);

sub run { system(@_) == 0 or confess("FAILED: ". join(" ", @_)); }

sub get_sorted_features {
    my ($ref_fasta, $ref_gff) = @_;
    my @seqs = read_fasta($ref_fasta);
    my %seqH = map { $_->[0] => $_->[2] } @seqs;
    my $gff = read_gff_tree($ref_gff);

    my %features;

    # assume gff is sorted
    for (@$gff) {
        my $contig = $_->{contig};
        my $start  = $_->{start};
        my $end    = $_->{end};
        my $strand = $_->{strand};
        my $length = $_->{length};
        my $func   = $_->{attribute}->{Name} || $_->{attribute}->{product};
        my $locus  = $_->{attribute}->{locus_tag};
        my $alias  = { LocusTag => $locus, GENE => $_->{attribute}->{Name} };
        my $desc   = $_->{descendants}->[0];
        $func      = $desc->{attribute}->{product} if $desc;
        my $note   = $desc->{attribute}->{Note} if $desc;
        $note    ||= $_->{attribute}->{Name};

        my $dna = uc substr($seqH{$contig}, $start-1, $length);
        $dna = rev_comp($dna) if $strand eq '-';

        my $feature = [ $_->{id},
                        'H37Rv.G2.v3',
                        $contig,
                        $start,
                        $end,
                        $strand,
                        $length,
                        $note,  # originally location
                        $func,
                        $alias,
                        $dna ];

        push @{$features{$contig}}, $feature;
    }

    wantarray ? %features : \%features;
}

sub feature_info_for_position {
    my ($ctg, $pos, $features) = @_;

    my ($left, $right, @cover);

    my $index = binary_search_in_sorted_features($ctg, $pos, $features);

    $left = $features->{$ctg}->[$index] if defined($index) && $index >= 0;
    # print STDERR '$features = '. Dumper($features);

    my @cover;
    for (my $i = $index + 1; $i < @{$features->{$ctg}}; $i++) {
        my $fea = $features->{$ctg}->[$i];
        my ($lo, $hi) = @{$fea}[3, 4];
        my $overlap = $lo <= $pos && $pos <= $hi;
        if (! $overlap) { $right = $fea; last; }
        push @cover, $fea;
    }

    my @overlapping_genes = sort { $b->[6] <=> $a->[6] } @cover; # sort genes by length
    # my @overlapping_other = grep { $_->[0] !~ /peg/ } @cover;
    my @overlapping_other;

    my $frameshift = 0;
    my (%func_seen, %func_cnt, %func_lo, %func_hi);
    for (@overlapping_genes) {
        my $len  = $_->[6];
        my $func = $_->[8];
        my $lo   = $_->[3];
        my $hi   = $_->[4];
        $frameshift = 1 if $len % 3;
        if (!hypo_or_mobile($func)) {
            $func_cnt{$func}++;
            $func_lo{$func} = $lo if $lo < $func_lo{$func} || ! defined($func_lo{$func});
            $func_hi{$func} = $hi if $hi > $func_hi{$func};
        }
    }
    for (($left, $right)) {
        my $func = $_->[8];
        my $lo   = $_->[3];
        my $hi   = $_->[4];
        if ($func_cnt{$func}) {
            $func_cnt{$func}++;
            $func_lo{$func} = $lo if $lo < $func_lo{$func} || ! defined($func_lo{$func});
            $func_hi{$func} = $hi if $hi > $func_hi{$func};
        }
    }
    for my $func (keys %func_cnt) {
        my $len = $func_hi{$func} - $func_lo{$func} + 1;
        $frameshift = 1 if $func_cnt{$func} > 1 && $len % 3;
    }

    my ($gene) = @overlapping_genes;
    my $pos_in_gene;
    if ($gene) {
        my ($lo, $hi, $strand) = @{$gene}[3, 4, 5];
        $pos_in_gene =  $strand eq '+' ? $pos - $lo + 1 : $hi - $pos + 1;
    }

    my %hash;

    $hash{overlapping_genes} = \@overlapping_genes if @overlapping_genes;
    $hash{overlapping_other} = \@overlapping_other if @overlapping_other;
    $hash{gene}              = $gene               if $gene;
    $hash{pos_in_gene}       = $pos_in_gene        if $pos_in_gene;
    $hash{frameshift_region} = $frameshift         if $frameshift;
    $hash{left}              = $left               if $left;
    $hash{right}             = $right              if $right;

    wantarray ? %hash : \%hash;
}

# Assumes the features are in ascending order on left coordinate and then right coordinate.
# Find the index of the rightmost feature who does not have a right neighbor that is to the left of the position
# Return -1 if no such feature can be found.

sub binary_search_in_sorted_features {
    my ($ctg, $pos, $features, $x, $y) = @_;

    return    if !$features || !$features->{$ctg};
    return -1 if $features->{$ctg}->[0]->[4] >= $pos;

    my $feas = $features->{$ctg};
    my $n = @$feas;

    $x = 0      unless defined $x;
    $y = $n - 1 unless defined $y;

    while ($x < $y) {
        my $m = int(($x + $y) / 2);

        # Terminate if:
        #   (1) features[m] is to the left pos, and
        #   (2) features[m+1] covers or is to the right of pos

        my $m2 = $feas->[$m]->[4];
        my $n2 = $feas->[$m+1]->[4];

        return $m if $m2 < $pos && ($n2 >= $pos || !defined($n2));

        if ($m2 < $pos) { $x = $m + 1 } else { $y = $m }
    }

    return $x;
}


sub read_gff_tree {
    my ($file) = @_;
    my $header = `cat $file | grep "^#"`;
    my @lines = `cat $file | grep -v "^#"`;
    # my @lines = `cat $file | grep -v "^#" |head`;
    my %id_to_index;
    my %rootH;
    my @features;
    my $index;
    shift @lines if $lines[0] =~ /region/;
    for (@lines) {
        chomp;
        my ($contig, $source, $feature, $start, $end, $score, $strand, $fname, $attribute) = split /\t/;
        my %hash = map { my ($k,$v) = split /=/; $k => $v } split(/;\s*/, $attribute);
        my $id = $hash{ID};
        my $ent = { id => $id,
                    contig => $contig,
                    source => $source,
                    feature => $feature,
                    start => $start,
                    end => $end,
                    length => $end - $start + 1,
                    score => $end,
                    strand => $strand,
                    fname => $fname,
                    attribute => \%hash };
        my $parent = $hash{Parent};
        if (!$parent) {
            push @features, $ent;
            $id_to_index{$id} = $index++;
            next;
        }
        while ($parent) {
            $rootH{$id} = $parent;
            $parent = $rootH{$parent};
        }
        my $root_index = $id_to_index{$rootH{$id}};
        push @{$features[$root_index]->{descendants}}, $ent;
    }
    return \@features;
}

sub read_fasta
{
    my $dataR = ( $_[0] && ref $_[0] eq 'SCALAR' ) ?  $_[0] : slurp( @_ );
    $dataR && $$dataR or return wantarray ? () : [];

    my $is_fasta = $$dataR =~ m/^[\s\r]*>/;
    my @seqs = map { $_->[2] =~ tr/ \n\r\t//d; $_ }
               map { /^(\S+)([ \t]+([^\n\r]+)?)?[\n\r]+(.*)$/s ? [ $1, $3 || '', $4 || '' ] : () }
               split /[\n\r]+>[ \t]*/m, $$dataR;

    #  Fix the first sequence, if necessary
    if ( @seqs )
    {
        if ( $is_fasta )
        {
            $seqs[0]->[0] =~ s/^>//;  # remove > if present
        }
        elsif ( @seqs == 1 )
        {
            $seqs[0]->[1] =~ s/\s+//g;
            @{ $seqs[0] } = ( 'raw_seq', '', join( '', @{$seqs[0]} ) );
        }
        else  #  First sequence is not fasta, but others are!  Throw it away.
        {
            shift @seqs;
        }
    }

    wantarray() ? @seqs : \@seqs;
}

sub slurp
{
    my ( $fh, $close );
    if ( $_[0] && ref $_[0] eq 'GLOB' )
    {
        $fh = shift;
    }
    elsif ( $_[0] && ! ref $_[0] )
    {
        my $file = shift;
        if    ( -f $file                       ) { }
        elsif (    $file =~ /^<(.*)$/ && -f $1 ) { $file = $1 }  # Explicit read
        else                                     { return undef }
        open( $fh, '<', $file ) or return undef;
        $close = 1;
    }
    else
    {
        $fh = \*STDIN;
        $close = 0;
    }

    my $out = '';
    my $inc = 1048576;
    my $end =       0;
    my $read;
    while ( $read = read( $fh, $out, $inc, $end ) ) { $end += $read }
    close( $fh ) if $close;

    \$out;
}

sub rev_comp {
    my ($dna) = @_;
    $dna = reverse($dna);
    $dna =~ tr/acgtumrwsykbdhvACGTUMRWSYKBDHV/tgcaakywsrmvhdbTGCAAKYWSRMVHDB/;
    return $dna;
}

sub hypo_or_mobile {
    my ($func) = @_;
    return 0;
    # return !$func || SeedUtils::hypo($func) || $func =~ /mobile/i;
}
