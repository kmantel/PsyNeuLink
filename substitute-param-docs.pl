# use File::Slurper;
use strict;
use warnings;

my ($fname, $class_name, $sub_str) = @ARGV;

rename($fname, $fname.'.orig');

# my $text = read_text($file.'orig');
# open(my $sub_fh, '<', $sub_fname) or die $!;
open(IN, '<'.$fname.'.orig') or die $!;
open(OUT, '>'.$fname) or die $!;

my $str = do {local $/; <IN>};
# my $sub_str = do {local $/; <$sub_fh>};

# $str =~ s/(class Parameters\([^\n]*?)\n(#? *)([^"\n]*)""".*?"""/$1\n$sub_str/igs;
$str =~ s/(class $class_name\(.*?)(class Parameters\([^\n]*?)\n(#? *)([^"\n]*)""".*?"""/$1$2\n$sub_str/igs;

print OUT $str;

