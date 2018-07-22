  git clone https://github.com/msharmavikram/faiss vsm2_faiss
  cd vsm2_faiss/
  ./configure
  echo "Remove CPU Flags"
  vi makefile.inc
  make 
  make install 
  export LD_LIBRARY_PATH=/home/vsm2/vsm2_faiss/
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/vsm2/vsm2_faiss/
  export PATH=$PATH:/home/vsm2/vsm2_faiss/
  cd demos/
  make demo_ivfpq_indexing 
  ./demo_ivfpq_indexing 
  cd ..
  make py 
  cd gpu/
  make 
  cd ../python/
  make _swigfaiss_gpu.so 
  make install
  export PYTHONPATH=$PYNTHONPATH:/home/vsm2/vsm2_faiss/python/
  python -c "import faiss"
  python -c "import _swigfaiss_gpu"
  cd ..
  cd demos/
  wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar -xvf sift.tar.gz 
  rm -rf sift.tar.gz 
  mv sift sift1M
  python demo_ondisk_ivf.py 0
  python demo_ondisk_ivf.py 1
  python demo_ondisk_ivf.py 2
  python demo_ondisk_ivf.py 3
  python demo_ondisk_ivf.py 4
  python demo_ondisk_ivf.py 5
  python demo_ondisk_ivf.py 6
