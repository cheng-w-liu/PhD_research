#include <iostream> 
#include <fstream>
#include <math.h>
#include <iomanip>
#include <cassert>
using namespace std; 

const int nsample = 1000;

int dd, ll, nn;
int npnt, nbin, nline;

double *p_table;
double *ss_table;
double **data;   
double *enrg; 
double *maga, *mag2, *mag4;
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
  int n, i;
  double av1, av2, av3, av4;
  double er1, er2, er3, er4;
  ofstream resfl;

  initran();

  readinput(); 
  readdata();

  resfl.open("res.dat", ios::trunc);
  for (n = 0; n < npnt; ++n) {
    for (i = 0; i < nbin; ++i) {
      enrg[i] = data[ n + i * npnt ][0];
      maga[i] = data[ n + i * npnt ][1];
      mag2[i] = data[ n + i * npnt ][2];
      mag4[i] = data[ n + i * npnt ][3];
      chi[i] = ( mag2[i] - pow(maga[i],2) );
      binder[i] = 0.5e0 * ( 3.0e0 - mag4[i]/pow(mag2[i],2) ) ;
    }

    bootstrapping(enrg, nbin, nsample, av1, er1); //av1 *= double(nn); er1 *= double(nn);
    bootstrapping(mag2, nbin, nsample, av2, er2);
    bootstrapping(chi, nbin, nsample, av3, er3); av3 *= double(nn); er3 *= double(nn);
    bootstrapping(binder, nbin, nsample, av4, er4);

    resfl << setw(20) << setprecision(14) << fixed << p_table[n]
	  << setw(20) << setprecision(14) << ss_table[n]
	  << setw(22) << setprecision(14) << av1
	  << setw(22) << setprecision(14) << er1 
	  << setw(22) << setprecision(14) << av2
	  << setw(22) << setprecision(14) << er2
	  << setw(22) << setprecision(14) << av3
	  << setw(22) << setprecision(14) << er3
	  << setw(22) << setprecision(14) << av4
	  << setw(22) << setprecision(14) << er4
	  << endl;

  } // end of n-loop
  resfl.close();

  for (i = 0; i < nline; ++i) delete [] data[i]; delete [] data;
  delete [] enrg;
  delete [] maga;
  delete [] mag2;
  delete [] mag4;
  delete [] chi;
  delete [] binder;
  delete [] p_table;
  delete [] ss_table;

  return 0; 
}//end of 'main' 




/*----------------------------------*/
void readdata(void){
/*----------------------------------*/
 fstream datafile; 
 int i;  
 double tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
    
 datafile.open("data.dat"); 
 i = 0;
 if ( datafile.is_open() ) {
   while ( datafile >> tmp1 >> tmp2 >> tmp3 >> tmp4 >> tmp5 >> tmp6 ) ++i;
 } else {
   cout << "can't open file data.dat" << endl; 
   exit (1);
 }
 datafile.close();  

 nline = i;
 if (nline%npnt != 0) {
   cout << "# of data is WRONG! \n";
   exit (1);
 }

 nbin = nline/npnt;

 data = new double *[nline];
 for (i = 0; i < nline; ++i) data[i] = new double [4];

 enrg = new double [nbin];
 maga = new double [nbin];
 mag2 = new double [nbin];
 mag4 = new double [nbin];
 chi = new double [nbin];
 binder = new double [nbin];

 datafile.open("data.dat");
 for ( i = 0; i < nline; ++i ) {
   datafile >> tmp1 >> tmp2 >> data[i][0] >> data[i][1] >> data[i][2] >> data[i][3]; 
 }
 datafile.close();

 p_table = new double [npnt];
 ss_table = new double [npnt];

 datafile.open("data.dat");
 for ( i = 0; i < npnt; ++i ) { 
   datafile >> p_table[i] >> ss_table[i] >> tmp3 >> tmp4 >> tmp5 >> tmp6;
 }
 datafile.close();

}//end of function 'readdata'



/*----------------------------------*/
void readinput(void){
/*----------------------------------*/
/*  Read file input.in,
    which has the format:
      
       dd ll h 
       mm 
       init istp nmsr nbins
       npnt 
*/
/*----------------------------------*/
  ifstream inputfile ("input.in");
  int mm, init, istp, mstp, nbn;

  if ( inputfile.is_open() ){
    while ( ! inputfile.eof() ){
      inputfile >> dd >> ll;
      inputfile >> mm >> npnt;
      inputfile >> init >> istp >> mstp >> nbn;
    }
  } else {
    cout << "Unable to open input.in" << endl;
    exit (1);
  }
  inputfile.close();

  nn = int(pow(ll,dd));

}//end of function 'readinput'



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
