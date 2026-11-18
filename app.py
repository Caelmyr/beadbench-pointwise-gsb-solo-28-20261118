import os
from image_lab import create_app
from beadbench.server import install
from waitress import serve
root=os.environ.get('BEADBENCH_DATA_DIR','data')
app=create_app(os.path.join(root,'preparation'))
install(app,os.path.join(root,'beads'))
if __name__=='__main__':serve(app,host='0.0.0.0',port=int(os.environ.get('PORT','18931')),threads=4)
