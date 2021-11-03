# clustering_embeddings
How to cluster embedding to minimize communication on a distributed infrastructure 

1. This clustering algorithm requires a library called metis. We must first instal this c library. Instillation only requires GNU make and CMAKE
- download the tar file from here: http://glaros.dtc.umn.edu/gkhome/metis/metis/download
gunzip metis-5.x.y.tar.gz
tar -xvf metis-5.x.y.tar
cd metis-5.x.y
make config shared=1 prefix=~path-to-install-library
make
make install 
hopefully it worked :)

2. install pymetis
pip install pymetis 

3. Should be good to run this script: python3 ./metis_main_vanilla.py