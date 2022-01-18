#!/bin/bash

echo "Downloading dataset..."
gdown --id 17gAy-P0EgmJHPespz3a01PZQiEhsH3jv -o dota.zip

echo "Extracting dataset..."
unzip dota.zip -d ../dataset/

echo "Done!"