#/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M IMI2x12,PQ32 nprobe=128
#/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M IMI2x12,PQ16 nprobe=128
#/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M IVF4096,PQ32 nprobe=128
#/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M IVF4096,PQ16 nprobe=128

/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M OPQ8_32,IVF4096,PQ8 nprobe=128
/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M OPQ8_32,IVF4096,PQ64 nprobe=128
/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M OPQ8_32,IMI2x12,PQ8 nprobe=128
/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M OPQ8_32,IMI2x12,PQ64 nprobe=128
/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M OPQ8_8,IVF4096,PQ64 nprobe=128
/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M OPQ8_8,IMI2x12,PQ64 nprobe=128
/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M OPQ8_16,IVF4096,PQ64 nprobe=128
/usr/bin/time -v python bench_polysemous_1bn.py SIFT5M OPQ8_16,IMI2x12,PQ64 nprobe=128
