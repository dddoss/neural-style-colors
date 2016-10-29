if [ -z "$PS1" ]; then echo -e "This script must be sourced. Use \"source ./setup.sh\" instead." ; exit ; fi

if [ ! -e ./imagenet-vgg-verydeep-19.mat ]; then
    wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat;
fi

if [ ! -e ./venv/ ]; then
    sudo pip install virtualenv;
    virtualenv venv;
fi

source ./venv/bin/activate;
pip install -r ./requirements.txt;
