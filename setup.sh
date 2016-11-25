if [ -z "$PS1" ]; then echo -e "This script must be sourced. Use \"source ./setup.sh\" instead." ; exit ; fi

if [ ! -e ./vgg_model/vgg19_data.npy ]; then
    wget people.csail.mit.edu/ddoss/vgg19/tf/vgg19_data.npy \
        -O ./vgg_model/vgg19_data.npy;
fi

if [ ! -e ./venv/ ]; then
    sudo pip install virtualenv;
    virtualenv venv;
fi

source ./venv/bin/activate;
sudo pip install -q -r ./requirements.txt;

python ./vgg_model/preprocess_vggdata.py
