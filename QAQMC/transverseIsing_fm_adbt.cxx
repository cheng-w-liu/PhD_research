#include <iostream>
#include <fstream>
#include "qaqmc.h"

using namespace cwliu_QAQMC_namespace;

void check_stop();

int main() {
  QAQMC qmc;
  int istp, mstp, nbins;

  qmc.initialize();
  qmc.simulation_parameters(istp,mstp,nbins);

  for ( int i = 0; i < istp; ++i) 
    qmc.mc_sweep();
  if (istp>0) qmc.write_conf();
  check_stop();

  for ( int i = 0; i < nbins; ++i) {
    qmc.clean();
    for ( int j = 0; j < mstp; ++j) {
      qmc.mc_sweep();
      qmc.measure();
    }
    qmc.write_data();
    qmc.write_conf();
    check_stop();
  }

  return 0;
}// end of main


void check_stop()
{
  std::ifstream oldstopfile;
  std::ofstream newstopfile;
  int i;

  oldstopfile.open("stop.txt");
  if ( oldstopfile.is_open() ) {
    oldstopfile >> i;
    oldstopfile.close();
  } else {
    std::cout << "Unable to open stop.txt \n";
  }

  if (i != 0) {
    newstopfile.open("stop.txt", std::ios::trunc);
    newstopfile << "stopped" << std::endl;
    newstopfile.close();
    exit (0);
  }
} // check_stop
