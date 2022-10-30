#!/bin/bash
# We start by taking the first 10 IDs of the descriptions longer than 100 characters
# each of the following steps will pipe its result into the next one
	# cat reads the content of the file
	# cut takes only the 2nd and 8th column of the file (sid_profile, description)
	# grep selects only the columns with longer descriptions than 100
	# head takes only the first 10 results
	# cut takes only the first column (sid_profile)
ids=$(cat instagram_posts.csv| cut --fields=2,8 -s | grep -oE '^-?[[:digit:]]+[[:blank:]]+.{100,}$' | head -10 | cut -f1)

# then we iterate through the found IDs:
	# if the ID is -1: 
		# then there is no corresponding user, we print "User was not Found"
	# else:
		# cat reads the content of the profiles file
		# cut takes only 1st and 3rd column (sid, username)
		# grep selects only the line where there is the searched ID
		# cat prints out the result

while IFS= read -r id; do
	if [ "$id" = -1 ]; then
		echo User was not Found
	else
		cat instagram_profiles.csv | cut -f 1,3 | grep -w "$id" | cat
	fi
done <<< "$ids"


