#!/usr/bin/env bash

if [ -n $1 ]; then
    fname='tmp'
else
    fname=$1
fi

cp "$fname" "$fname".tmp

sed -i 's;params = self._assign_args_to_param_dicts(;;g' "$fname".tmp
sed -i 's;)$;,;g' "$fname".tmp
sed -i 's; ;;g' "$fname".tmp
# sed -i 's;=.*$;;g' "$fname".tmp
sed -i 's;\n; ;g' "$fname".tmp
tr '\n' ' ' < "$fname".tmp

rm "$fname".tmp

echo
