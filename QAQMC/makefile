### suffix rule
.SUFFIXES: .cxx .o


### link objects and compile
qmc.out: transverseIsing_fm_adbt.o qaqmc.o
	g++ -O3 transverseIsing_fm_adbt.o qaqmc.o -o qmc.out


### create objects 
qaqmc.o:	qaqmc.cxx
	g++ -O3 -Wall -c qaqmc.cxx


transverseIsing_fm_adbt.o: transverseIsing_fm_adbt.cxx
	g++ -O3 -Wall -c transverseIsing_fm_adbt.cxx


### clean
clean: 
	rm -f *.o
	rm -f *.out

### dependencies
transverseIsing_fm_adbt.o: qaqmc.h
qaqmc.o: qaqmc.h

