# !/bin/bash

# push TinyVLM
echo "--------------------------"
echo "update TinyVLM codes..."
git add .
git commit -m "update"
git pull
git push


# push utils
echo "--------------------------"
echo "update utils codes..."
cd utils
git add .
git commit -m "update"
git pull
git push
