



echo "Using `bash run.sh` to run this file."


pip3 install torch tqdm


nohup python main.py > stress_test_results.log &


echo "Process is now on the background."
