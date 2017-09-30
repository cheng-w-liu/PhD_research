#include <stdlib.h>
#include <iostream> 
#include <fstream>
#include <math.h>
#include <iomanip>
#include <cassert>
using namespace std; 

const int nsample = 1000;

int D, L, N; 
double T, beta; 
int nbin;

double **data;   
double *enrg; 
double *ma, *m2, *m4;
double *chi, *binder;

// variables for Random Numer Generator
int iir, jjr, kkr, nnr, seed[4];

void readinput(void);
void readdata(void);
void bootstrapping(double vec[], int n, int nboot, double& ave, double& err);
void initran(void);
double ran(void);

void aveanderr(double [],int ,double &, double &);
void aveanderr(long double [],int ,long double &, long double &);

   
int main () { 
  int i, b;
  double av1, av2, av3, av4;
  double er1, er2, er3, er4;

  initran();

  readinput();
  readdata();
  
  aveanderr(enrg, nbin, av1, er1);
  aveanderr(m2, nbin, av2, er2);
 
  bootstrapping(chi, nbin, nsample, av3, er3);
  bootstrapping(binder, nbin, nsample, av4, er4);

  cout << setw(12)  << setprecision(8) << fixed << T
       << setw(20) << setprecision(12) << av1
       << setw(18) << setprecision(12) << er1 
       << setw(18) << setprecision(12) << av2
       << setw(18) << setprecision(12) << er2
       << setw(18) << setprecision(12) << av3
       << setw(18) << setprecision(12) << er3
       << setw(18) << setprecision(12) << av4
       << setw(18) << setprecision(12) << er4
       << endl;

  delete [] enrg;
  delete [] ma;
  delete [] m2;
  delete [] m4;
  delete [] chi;
  delete [] binder;

  return 0; 
}//end of 'main' 



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

}



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



/*------------------------------------*/
void aveanderr(long double vec[], int n,long double& ave,long double& err){
/*------------------------------------*/

 int i;
 
 ave=0.0e0; err=0.0e0;
 for ( i=0; i<n; i++ ) {
       ave+=vec[i];
       err+=vec[i]*vec[i];
 }

 ave/=double(n);
 err/=double(n);
 err = sqrt( fabs(err-ave*ave)/double(n-1) ) ; 
}//end of 'aveanderr'



/*----------------------------------*/
void readdata(void){
/*----------------------------------*/
 fstream datafile; 
 int i;
 double tmp1, tmp2, tmp3, tmp4;
    
 datafile.open("data.dat"); 
 i = 0;
 if ( datafile.is_open() ) {
   while ( datafile >> tmp1 >> tmp2 >> tmp3 >> tmp4 ) ++i;
 } else {
   cout << "can't open file data.dat" << endl; 
   exit (1);
 }
 datafile.close();  

 nbin = i;

 //data = new double *[nbin];
 //for (i = 0; i < nbin; ++i) data[i] = new double [4];

 enrg = new double [nbin];
 ma = new double [nbin];
 m2 = new double [nbin];
 m4 = new double [nbin];
 chi = new double [nbin];
 binder = new double [nbin];

 datafile.open("data.dat");
 for ( i = 0; i < nbin; ++i ) {
   datafile >> enrg[i] >> ma[i] >> m2[i] >> m4[i]; 
   chi[i] = double(N) * beta * ( m2[i] - ma[i]*ma[i] );
   binder[i] = 0.5 * (3.0 - m4[i]/(m2[i]*m2[i]));
 }
 datafile.close();

}//end of function 'readdata'



/*----------------------------------*/
void readinput(void){
/*----------------------------------*/
//  Read file input.in, 
//    which has the format
//       D L T
//       init istp mstp nbin 
/*----------------------------------*/
  ifstream inputfile ("input.in");
  int init, istp, mstp, nbn; 

  if ( inputfile.is_open() ){
    while ( !inputfile.eof() ){
      inputfile >> D >> L >> T;
      inputfile >> init >> istp >> mstp >> nbn;
    }
  } else {
    cout << "Unable to open input.in" << endl;
    exit (1);
  }
  inputfile.close();

  N = int(pow(L,D));
  beta = 1.0/T;
  
}//end of function 'readinput' 



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
}// end of member function 'Cadbt_qmc::initran'


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
}// end of member function 'Cadbt_qmc::ran'
