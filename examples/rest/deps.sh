#!/bin/bash
# Download and update deps for binary

# get the directory of this script file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PUBLIC=$DIR/public

#echo "download js bundle files"
#curl https://npm.reversehttp.com/@preact/signals-core,@preact/signals,htm/preact,preact,preact/hooks > $PUBLIC/index.js
#echo >> $PUBLIC/index.js # add newline

FILES=$(find "${PUBLIC}" -type f ! -iname "*.hpp")

echo -n "" > "${DIR}/public.hpp"
echo "#ifndef LLAMA_CPP_REST_PUBLIC_HPP" >> "${DIR}/public.hpp"
echo "#define LLAMA_CPP_REST_PUBLIC_HPP" >> "${DIR}/public.hpp"

cd $PUBLIC
for FILE in $FILES; do
  EXT=$(echo -n ${FILE: -5} | tr '[:upper:]' '[:lower:]')
  if [ $EXT == ".html" ]; then
    echo "generate ${FILE}.hpp"
    BFILE=$(basename "${FILE}")
    NFILE=$(echo -n "${BFILE}" | tr "." "_")
    xxd -n "${NFILE}" -i "${FILE}" > "${PUBLIC}/${BFILE}.hpp"

    echo "#include \"public/${BFILE}.hpp\"" >> "${DIR}/public.hpp"
  else
    echo "skip     ${FILE}"
  fi

done

echo "#endif" >> "${DIR}/public.hpp"