ENV_NAME='rocksample_11_11'

rsync -zLurP --exclude "*/*/${ENV_NAME}_seed*/" pedro:"/home/taodav/Documents/pobax/results/$ENV_NAME" ../results/