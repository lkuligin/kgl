sudo apt-get install -y python
sudo apt-get update
sudo apt-get install -y python-pip
sudo pip install virtualenv
virtualenv -p python3 py3
source ./py3/bin/activate
pip install pandas==0.19
pip install numpy
pip install scipy
pip install sklearn
pip install xgboost
#sudo apt-get install python3-dev
#sudo apt-get install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev