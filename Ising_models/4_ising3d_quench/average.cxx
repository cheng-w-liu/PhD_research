#include <stdlib.h>
#include <iostream> 
#include <fstream>
#include <math.h>
#include <iomanip>
#include <cassert>
using namespace std; 

int nbins, nline;
int tau, qpnt;
int tot_pnt; // tot_pnt = qpnt + 1 

int Nboot = 1000;

int *tm_table; // time table, t
double *tmpt_table; // temperature table, T(t)
double **data;   
double *energy; 
double *maga;
double *mag2;
double *mag4;
double *binder;

// variables for Random Numer Generator
int iir, jjr, kkr, nnr, seed[4];

void initran(void);
double ran(void);

void read_input(void);
void read_data(void);
void aveanderr(double [],int ,double &, double &);
void bootstrapping(double vec[], int n, int nboot, double& ave, double& err);
   
int main () { 
  double av1, av2, av3;
  double er1, er2, er3;
  ofstream datafl;

  read_input();
  read_data();

  datafl.open("res.dat", ios::trunc);
  for (int t = 0; t < tot_pnt; ++t) {

    for (int i = 0; i < nbins; ++i) {      
      energy[i] = data[ t + i * tot_pnt ][0];
      maga[i]   = data[ t + i * tot_pnt ][1]; // not used
      mag2[i]   = data[ t + i * tot_pnt ][2];
      mag4[i]   = data[ t + i * tot_pnt ][3];
      binder[i] = 0.5e0 * ( 3.0e0 - mag4[i]/pow(mag2[i],2.0) ) ;
    }

    aveanderr(energy, nbins, av1, er1);
    aveanderr(mag2, nbins, av2, er2);
    bootstrapping(binder, nbins, Nboot, av3, er3);

    datafl << setw(12) << fixed << tm_table[t] 
	   << setw(24) << setprecision(14) << tmpt_table[t]
	   << setw(22) << setprecision(14) << av1
	   << setw(22) << setprecision(14) << er1 
	   << setw(22) << setprecision(14) << av2
	   << setw(22) << setprecision(14) << er2
	   << setw(22) << setprecision(14) << av3
	   << setw(22) << setprecision(14) << er3
	   << endl;

  } // end of t-loop      
  datafl.close();

  for (int i = 0; i < nline; ++i) delete [] data[i]; 
  delete [] data;
  delete [] energy;
  delete [] maga;
  delete [] mag2;
  delete [] mag4;
  delete [] binder;
  delete [] tm_table;
  delete [] tmpt_table;

  return 0; 
}//end of 'main' 




/*------------------------------------*/
void aveanderr(double vec[], int n, double& ave, double& err){
/*------------------------------------*/

 int i;
 
 ave = 0.0e0; err = 0.0e0;
 for ( i = 0; i < n; ++i ) {
       ave += vec[i];
       err += vec[i]*vec[i];
 }

 ave/=double(n);
 err/=double(n);
 err = sqrt( (err-ave*ave)/double(n-1) ) ; 
}//end of 'aveanderr'






/*----------------------------------*/
void read_data(void){
/*----------------------------------*/
 fstream datafile; 
 int i, tmp0;  
 double tmp1, tmp2, tmp3, tmp4, tmp5;
    
 datafile.open("data.dat"); 
 i = 0;
 if ( datafile.is_open() ) {
   while ( datafile >> tmp0 >> tmp1 >> tmp2 >> tmp3 >> tmp4 >> tmp5 ) ++i;
 } else {
   cout << "can't open file data.dat" << endl; 
   exit (1);
 }
 datafile.close();  

 nline = i;
 if (nline%tot_pnt != 0) {
   cout << "# of data is WRONG! \n";
   exit (1);
 }

 nbins = nline/tot_pnt;

 data = new double *[nline];
 for (i = 0; i < nline; ++i) data[i] = new double [4];

 tm_table = new int [tot_pnt];
 tmpt_table = new double [tot_pnt];

 energy = new double [nbins];
 maga = new double [nbins];
 mag2 = new double [nbins];
 mag4 = new double [nbins];
 binder = new double [nbins];

 datafile.open("data.dat");
 for ( i = 0; i < nline; ++i ) {
   datafile >> tmp0 >> tmp1 >> data[i][0] >> data[i][1] >> data[i][2] >> data[i][3];
 }
 datafile.close();

 datafile.open("data.dat");
 for ( i = 0; i < tot_pnt; ++i ) {
   datafile >> tm_table[i] >> tmpt_table[i] >> tmp2 >> tmp3 >> tmp4 >> tmp5;
 }
 datafile.close();

}//end of function 'read_data'



/*----------------------------------*/
void read_input(void){
/*----------------------------------*/
//  Read file input.in, 
//    which has the format
//       L
//       tau
//       init istp mstp nbins 
/*----------------------------------*/
  ifstream inputfile ("input.in");
  int L;
  int init, istp, nbn, mstp; 

  if ( inputfile.is_open() ){
    while ( !inputfile.eof() ){
      inputfile >> L;
      inputfile >> tau >> qpnt;
      inputfile >> init >> istp >> mstp >> nbn;
    }
  } else {
    cout << "Unable to open input.in" << endl;
    exit (1);
  }
  inputfile.close();

  assert( tau%qpnt == 0 );

  tot_pnt = qpnt + 1;


}//end of function 'read_input' 



/*------------------------------------*/
void bootstrapping(double vec[], int n, int nboot, double& ave, double& err)
/*------------------------------------*/
{
  int i, j;
  double sum;

  ave = 0.0e0;
  for (i = 0; i < n; ++i) ave += vec[i];
  ave /= double(n);

  err = 0.0e0;
  for ( j = 0; j < nboot; ++j) {
    sum = 0.0e0;
    for ( i = 0; i < n; ++i) sum += vec[ int(ran()*n) ];
    sum /= double(n);
    err += (sum - ave)*(sum - ave);
  }
  err = sqrt(err/double(nboot));

} // bootstrapping




double ran(void) {
  int mzran;
  double r;

  mzran=iir-kkr;
  if(mzran<0) mzran+=2147483579;
  iir=jjr;
  jjr=kkr;
  kkr=mzran;
  nnr=69069*nnr+1013904243;
  mzran=mzran+nnr;
  r=0.5e0+double((0.23283064e-9)*mzran);

  return r;
} // ran



void initran(void) {
  ifstream seedfile ("seed.in");
  int i;

  if ( seedfile.is_open() ) {
    while ( ! seedfile.eof() ){
      for ( i=0; i<4; i++ ) seedfile >> seed[i];
    }
  } else {
    cout << "Unable to open the seed file" << endl;
    exit (1);
  } //end of 'if seedfile.is_open()'                                                                     
  seedfile.close();

  iir=1+int(fabs(seed[0]));
  jjr=1+int(fabs(seed[1]));
  kkr=1+int(fabs(seed[2]));
  nnr=seed[3];
} // initran()
