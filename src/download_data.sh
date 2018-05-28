# pip install kaggle-cli

kg download -u envikas -p crimson -c 'diabetic-retinopathy-detection'

# Create directories for test and train data
mkdir train
mkdir test

# Move ZIP files to their directories
mv train.* train
mv test.* test

# Extract data
7za x train.zip.001
7za x test.zip.001
