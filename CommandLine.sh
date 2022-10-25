#!/bin/bash
ids=$(cat instagram_posts.csv| cut --fields=2,8 -s | grep -oE '^[[:digit:]]+[[:blank:]]+.{100,}$' | head -10 | cut -f1 | mawk 8 ORS="\|")


cat instagram_profiles.csv | cut -f 1,3 | grep -w "${ids::-2}" | cat
