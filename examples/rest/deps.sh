#!/bin/bash
# Download and update deps for binary

# get the directory of this script file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HELP=$DIR/help

#echo "download js bundle files"
#curl https://npm.reversehttp.com/@preact/signals-core,@preact/signals,htm/preact,preact,preact/hooks > $PUBLIC/index.js
#echo >> $PUBLIC/index.js # add newline

FILES=$(find "${HELP}" -type f ! -iname "*.hpp")

cd $HELP
for FILE in $FILES; do
  echo "generate ${FILE}.hpp"

  # use simple flag for old version of xxd
  BFILE=$(basename "${FILE}")
  NFILE=$(echo -n "${BFILE}" | tr "." "_")
  xxd -n "${NFILE}" -i "${FILE}" > "${HELP}/${BFILE}.hpp"
done
