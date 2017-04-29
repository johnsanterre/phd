use Getopt::Long;

my $usage = "Usage: $0 [ options ] file \n\n";

my ($help, $k, $t, $ci, $tmp, $fm, $quiet);

GetOptions("h|help" => \$help,
           "c|ci=i" => \$ci,
           "k=i"    => \$k,
           "t=i"    => \$t,
           "tmp=s"  => \$tmp,
           "fm"     => \$fm,    # multililne fasta
           "q"      => \$quiet
          ) or die("Error in command line arguments\n");

$help and die $usage;
my $file = shift @ARGV or die $usage;

$ci  ||= 1;
$k   ||= 31;
$t   ||= 12;
$tmp ||= "$ENV{HOME}/tmp";
$fm  = "-fm" if $fm;

my $cs = 16777215;                 # max coverage
my $mem = 100;                  # memory in GB

run("mkdir -p $tmp");

my $out = "$file.k$k"; $out =~ s|.*/||;
my $in = is_sequence_file($file) ? $file : '@'.$file;
my $log = ">/dev/null" if $quiet;

my @cmd = ('/home/fangfang/bin/kmc', $fm, "-k$k", "-t$t", "-m$mem", "-ci$ci", "-cs$cs",
           $in, $out, $tmp, $log);

# print join(" ", @cmd) . "\n";

# run(@cmd);
run(join(" ", @cmd));
run("/home/fangfang/bin/kmc_dump $out $out.unsorted");
run("/home/fangfang/bin/sort -S 10% --parallel=$t $out.unsorted >$out");
# unlink("$out.kmc_suf");
# unlink("$out.kmc_pre");
unlink("$out.unsorted");

sub is_sequence_file {
    my ($file) = @_;
    return 1 if $file =~ /(fasta|fa|fna|fastq|fq)(\.gz|)?$/;
}

sub run { system(@_) == 0 or confess("FAILED: ". join(" ", @_)); }
