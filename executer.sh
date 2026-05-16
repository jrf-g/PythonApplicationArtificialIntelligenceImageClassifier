echo "STOP THIS SCRIPT IF YOU ARENT RUNNING FROM THE CLONED REPO"
echo "YOU HAVE TEN SECONDS TO STOP THE SCRIPT"
sleep 10
echo "TRAINING"
python optimizedcputrainer.py
echo "WAITING"
sleep 1
echo "PREDICTING"
python classifyandpredict.py
echo "FEEDBACK FORM AND PARTY PROGRAM LAUNCHING"
sleep 1
coffee -c congrats.coffee -o javascript/ -r readline --no-header
cd javascript
node congrats.js
echo "please wait. cleanup script."
cd -
echo "DONE"
read -p "DELETE TRAINING DATA? (requires rebuild of classify.png and samples. we only rebuild directories but not their contents) (y/n): " choice
if [[ "$choice" != "y" ]]; then
    echo "aborting in 10"
	sleep 10
    exit
fi
rm classify.png
rm -r samples
rm -r javascript
mkdir samples
mkdir javascript
echo "cleanup complete. please rebuild"
echo "aborting in 10"
sleep 10
exit
