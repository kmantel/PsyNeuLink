#!/usr/bin/env bash

indent="    "
outer="$indent$indent"
inner="$indent$indent$indent"

function get_indent {
    res=""
    for i in seq $1; do
        res="$res$indent"
    done
    echo $res
}

echo "$outer""super().__init__("
echo "$inner""default_variable=default_variable,"
for arg in $(get-args.sh); do
    echo "$inner""$arg"
done

for arg in $@; do
    if [[ *"$arg"* != "=" ]]; then
        arg="$arg=$arg"
    fi
    echo "$inner""$arg,"
done

printf "$outer"")"
