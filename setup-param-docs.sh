#!/usr/bin/env bash

# removes empty newlines directly after Parameters class definitions
perl -0777 -i.orig -pe 's;(class Parameters\([^\n]*?)\n( *\n)*;\1\n;igs' $1

# inserts docstring quotes for Parameters classes that do not currently have them
perl -0777 -i.orig -pe 's;(class Parameters\([^\n]*?)\n(#? *)([^"\n]*)\n;\1\n\2"""\n\2"""\n\2\3\n;igs' $1
