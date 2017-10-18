sudo apt-get install -y python
sudo apt-get update
sudo apt-get install -y python-pip
sudo pip install virtualenv
virtualenv -p python3 py3
source ./py3/bin/activate
pip install pandas
pip install numpy
pip install sklearn
pip install xgboost
