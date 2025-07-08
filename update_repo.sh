# !/bin/bash

# push TinyVLM
epch "--------------------------"
echo "update TinyVLM codes..."
git add .
git commit -m "update"
git pull
git push


# push utils
epch "--------------------------"
echo "update utils codes..."
cd utils
git add .
git commit -m "update"
git pull
git push
