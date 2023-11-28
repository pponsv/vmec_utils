.PHONY : test build all clean

MOD_NAME = vmec_helper

all: build

build: ./vmec_utils/helper/src/vmec_helper.f90 
	f2py -c --f90flags='-Wno-tabs -fopenmp -O2 -fPIC' -lgomp $< -m $(MOD_NAME)
	mv $(MOD_NAME).* ./vmec_utils/helper/

clean:
	rm -f ./helper/*.so

test:
	cd .. && python3 ./vmec_utils/test/test_booz.py
	cd .. && python3 ./vmec_utils/test/test_vmec.py

prof:
	scalene ./test/test_booz.py --profile-all --cpu --use-virtual-time --cpu-percent-threshold 0.1 --cpu-sampling-rate 0.001 --use-virtual-time True
