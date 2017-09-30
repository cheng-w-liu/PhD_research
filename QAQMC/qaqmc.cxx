// FILE: qaqmc.cxx
//
//
// 1) Definition of operator types
//                          o1    o2
//    H_{+1,b} = Ising(FM)  i(b)  j(b)
//    H_{-1,0} = h          -1    i
//    H_{-2,i} = Field      -2    i
//

#include "qaqmc.h"
#include <iostream>
#include <fstream> // provides 'ifstream', 'ofstream' datatype
#include <math.h>  // provides pow() and fabs() function
#include <iomanip> // provides 'setw()' function
#include <cassert> // Provides 'assert()'

namespace cwliu_QAQMC_namespace {

  void QAQMC::measure(void) 
  {    
    for (int i = 0; i < N; ++i) 
      spn2[i] = spn1[i];

    // ==================== 1st half ====================//
    for (int p = 0; p < MM/2; ++p) {
      
      // ---------- update configurations ----------
      if ( opstr[p].o1 == -2 ) spn2[ opstr[p].o2 ] *= -1;
      
      // ---------- measurement  points ----------
      if ( p_to_n.find(p) != p_to_n.end() ) 
	{	
	  int n = p_to_n[p];

	  if ( opstr[p].o1 == -1 ) 
	    ++et[n];
	  else if ( opstr[p].o1 == -2 ) 
	    ++ef[n];
	
	  int ising = 0;
	  for (int i = 0; i < nb; ++i) {
	    int s1 = bst[0][i];
	    int s2 = bst[1][i];
	    ising += spn2[s1] * spn2[s2];
	  }
	  ei[n] += ising;
	
	  int m = 0; 
	  for (int i = 0; i < N; ++i) m += spn2[i];

	  ma[n] += fabs(m)/double(N);
	  m2[n] += pow(double(m)/double(N),2);
	  m4[n] += pow(double(m)/double(N),4);

	} //end of 'if ( p_to_n.find(p) != p_to_n.end()  )'
      
    } // end of 'for (p=0; p<MM/2 ...)'


    // ==================== 2nd half ====================//
    for (int p = MM/2; p < MM; ++p) {

      // ---------- update configurations ----------
      if ( opstr[p].o1 == -2 ) spn2[ opstr[p].o2 ] *= -1;
      
      // ---------- measurement  points ----------
      if ( p_to_n.find(MM-2-p) != p_to_n.end() ) 
	{	
	  int n = p_to_n[MM-2-p];

	  if ( opstr[p].o1 == -1 ) 
	    ++et[n];
	  else if ( opstr[p].o1 == -2 ) 
	    ++ef[n];
	
	  int ising = 0;
	  for (int i = 0; i < nb; ++i) {
	    int s1 = bst[0][i];
	    int s2 = bst[1][i];
	    ising += spn2[s1] * spn2[s2];
	  }
	  ei[n] += ising;
	
	  int m = 0; 
	  for (int i = 0; i < N; ++i) m += spn2[i];

	  ma[n] += fabs(m)/double(N);
	  m2[n] += pow(double(m)/double(N),2);
	  m4[n] += pow(double(m)/double(N),4);

	} //end of 'if ( p_to_n.find(MM-2-p) != p_to_n.end()  )'

    } // end of 'for (p = MM/2; p < MM; ...)'

  } // end of member function 'QAQMC::measure'



  void QAQMC::write_data(void) 
  {
    int n, p, mstp1, mstp2 = 2*mstp;
    double jj, hh, enrg, eising, efield, maga, mag2, mag4;
    std::ofstream outfile("data.dat", std::ios::app);
    
    if ( outfile.is_open() ) {
      for ( n = 0; n < npnt-1; ++n ) {
	p = n_to_p[n];
	S = S_table[n];
	jj = S; 
	hh = (1.0e0 - S);

	eising = -jj*double(ei[n])/double(N*mstp2);
	if (et[n]==0) {
	  efield = 0.0;
	} else {
	  efield = -hh*double(ef[n])/double(et[n]);
	}
	enrg = eising + efield;

	maga = ma[n]/double(mstp2);
	mag2 = m2[n]/double(mstp2);
	mag4 = m4[n]/double(mstp2);    
		
	outfile << std::setw(20) << std::setprecision(14)  << std::fixed << double(p+1)/double(tau)
		<< std::setw(20) << std::setprecision(14)  << S
		<< std::setw(22) << std::setprecision(14)  << enrg
		<< std::setw(20) << std::setprecision(14)  << maga
		<< std::setw(20) << std::setprecision(14)  << mag2
		<< std::setw(20) << std::setprecision(14)  << mag4
		<< std::endl;
      } // end of 'for n = 0 ...'     


      if ( p_to_n.find(MM/2-1) != p_to_n.end() )
	mstp1 = mstp;
      else
	mstp1 = mstp2;

      n = npnt-1;
      p = n_to_p[n];
      S = S_table[n];
      jj = S; 
      hh = (1.0e0 - S);

      eising = -jj*double(ei[n])/double(N*mstp1);
      if (et[n]==0) {
	efield = 0.0;
      } else {
	efield = -hh*double(ef[n])/double(et[n]);
      }
      enrg = eising + efield;

      maga = ma[n]/double(mstp1);
      mag2 = m2[n]/double(mstp1);
      mag4 = m4[n]/double(mstp1);    
		
      outfile << std::setw(20) << std::setprecision(14)  << std::fixed << double(p+1)/double(tau)
	      << std::setw(20) << std::setprecision(14)  << S
	      << std::setw(22) << std::setprecision(14)  << enrg
	      << std::setw(20) << std::setprecision(14)  << maga
	      << std::setw(20) << std::setprecision(14)  << mag2
	      << std::setw(20) << std::setprecision(14)  << mag4
	      << std::endl;
      
    } else {
      std::cout << "can't open file data.dat" << std::endl;
      exit (1);
    } // end of 'if ( outfile.is_open() )'

  } // end of member function 'QAQMC::write_data'



  void QAQMC::clean(void) {

    for (int n = 0; n < npnt; ++n) {
      et[n] = ef[n] = ei[n] = 0;
      ma[n] = m2[n] = m4[n] = 0.0e0;
    }

  }// end of member function 'QAQMC::clean'



  void QAQMC::mc_sweep(void) {
    diagonal_update();
    vertices_link();
    cluster_update();    
  }// end of member function 'QAQMC::mc_sweep'



  void QAQMC::heatbath(void) {

    double jj, hh;

    jj= S; 
    hh = 1.0e0 - S;

    p0 = hh/( 2.0e0 * jj * double(dd) + hh );

  }// end of member function 'QAQMC::heatbath'



  void QAQMC::diagonal_update(void) {
    //----------------------------------//
    //  use heat bath probability to
    //  carry out the diagonal update
    //  H_{0,i} <-> H_{+1,b}
    //----------------------------------//

    for (int i=0; i<N; ++i) spn2[i] = spn1[i];

    S = Si;
    for (int p=0; p<MM; ++p) {

      if ( p < tau ) 
	quench_S(p+1);
      else if (p > tau )
	quench_S(p);

      heatbath();
      
      int o1 = opstr[p].o1;
      if ( o1 >= -1 ) { // H_{+1,b} or H_{-1,i}
	bool ok=false;
	while (!ok) 
	  {
	    if ( ran()<p0 ) { // accept --> H_{-1,i'}
	      opstr[p].o1 = -1; 
	      opstr[p].o2 = int( ran()*N );
	      ok=true;
	    } else { //try H_{+1,b}
	      int b = int( ran()*nb );
	      int s1 = bst[0][b];
	      int s2 = bst[1][b];
	      if ( spn2[s1]==spn2[s2] ) 
		{ // accept --> H_{+1,b}
		  opstr[p].o1 = s1;
		  opstr[p].o2 = s2;
		  ok=true;
		} //end of ' spn2[s1]==spn2[s2]'
	    } //end of 'if ( ran()<p0 )'
	  } //end of 'while (!ok)' 

      } else if ( o1 == -2 ) { //H_{-2,i}
	spn2[ opstr[p].o2 ] *= -1;
      } //end of 'if (o1>=-1)'

    } //end of 'for p=0...'

  }// end of member function 'QAQMC::diagonal_update'



  void QAQMC::vertices_link(void) {

    for (int i=0; i<N; ++i) 
      last[i] = frst[i] = -1;

    for (int i=0; i<4*MM; ++i)
      vrtx[i] = -1;

    for (int p=0; p<MM; ++p) {
      int o1 = opstr[p].o1;
      int o2 = opstr[p].o2;
      int v0 = 4*p;
      
      if ( o1<0 ) { // H_{0,i} or H_{-1,i} 

	if ( last[o2]==-1 ) { // o2 never been linked
	  frst[o2] = v0;
	} else { // o2 been linked before
	  vrtx[v0]=last[o2];
	  vrtx[ last[o2] ]=v0;
	}
	last[o2] = v0+2;

      } else { // H_{1,b}

	if ( last[o1]==-1 ) { // o1 never been linked
	  frst[o1] = v0;
	} else { // o1 been linked before 
	  vrtx[v0] = last[o1];
	  vrtx[ last[o1] ] = v0;
	}
	last[o1] = v0+2;

	if ( last[o2]==-1 ) { // o2 never been linked
	  frst[o2] = v0+1;
	} else { // o2 been linked before
	  vrtx[v0+1] = last[o2];
	  vrtx[ last[o2] ] = v0+1;
	}
	last[o2] = v0+3;

      } //end of 'if o1<0...'
    } //end of 'for p=0...'

    for (int i=0; i<N; ++i) {
      if ( frst[i]!=-1 ) { // site-i been linked 
	vrtx[ frst[i] ] = 4*MM+i;
	vrtx[ last[i] ] = 4*MM+N+i;
      } 
    } 
    
  } // end of member function 'QAQMC::vertices_link'


  
  void QAQMC::cluster_update(void) {
    //
    // H_{-1,i} <--> H_{-2,i}
    //

    cmark=0;

    //----- 1st cluster update, for internal vertices -----
    for (int i = 0; i<4*MM-1; i+=2) {
      int j=vrtx[i];
      int p=i/4;
      if ( opstr[p].o1>=0 || j<0 ) continue;
      --cmark;
      Stck.push(j);
      vrtx[i]=cmark;
      if ( ran()<0.5e0 ) {
	visit_cluster();
      } else {
	opstr[p].o1 = (opstr[p].o1==-1) ? -2 : -1 ;
	flip_cluster();
      } //end of 'if'
    } //end of 'for i=0'    


    //----- 2nd cluster update, for spn1[] edge vertices -----
    for (int i=0; i<N; ++i) {
      int j=frst[i];
      if ( j==-1 ) { // spin-j not been acted, i.e., free spin
	if ( ran() > 0.5e0) {
	  spn1[i]*=-1;
	  spn2[i]*=-1;
	}
      } else { // spin-j been acted, construct link 
	if ( vrtx[j]>=0 ) {
	  --cmark;
	  Stck.push(j);
	  if ( ran()<0.5e0 ) {
	    visit_cluster();
	  } else {
	    spn1[i]*=-1;
	    flip_cluster();
	  } //edn of 'if ran()<0.5'
	} //end of 'if vrtx[j]>0'
      } //end of 'if j==-1'                                                                                        
    } //end of 'for i=0...'


    for (int i=0; i<N; ++i) {
      int j=last[i];
      if ( j>=0 ) {
	if ( vrtx[j]>=0 ) {
	  --cmark;
	  Stck.push(j);
	  if ( ran()<0.5e0 ) {
	    visit_cluster();
	  } else {
	    spn2[i]*=-1;
	    flip_cluster();
	  } //edn of 'if ran()<0.5'
	} //end of 'if vrtx[j]>0'
      } //end of 'if j>=0'
    } //end of 'for i=0...'

  } // end of member function 'QAQMC::clusterupdate'



  void QAQMC::visit_cluster(void) {

    while ( !Stck.empty() ){
      int j = Stck.top(); Stck.pop();
      if ( j > 4*MM-1 ) continue;
      int p=j/4;
      vrtx[j]=cmark;
      if ( opstr[p].o1>=0 ) {
	for ( int k=4*p; k<4*p+4; ++k){
	  if ( vrtx[k]>=0 ) {
	    Stck.push(vrtx[k]);
	    vrtx[k] = cmark;
	  } //end of 'if vrtx[k]>0'
	} //end of 'for'
      } //end of 'if opstr[p].o1>=0'
    } //end of 'while (ns>0)...' 

  } // end of member function 'QAQMC::visit_cluster'



  void QAQMC::flip_cluster(void) {
    
    while ( !Stck.empty() ){
      int j = Stck.top(); Stck.pop();
      if ( j > 4*MM-1 ) {
	j-=4*MM;
	if ( j>N-1 ) { // links to spn2
	  j-=N;
	  spn2[j]*=-1;
	} else { // links to spn1
	  spn1[j]*=-1;
	} //end of 'j>N-1'
	continue;
      } //end of 'if vrtx[j]>4*MM-1'

      int p=j/4;
      vrtx[j]=cmark;
      if ( opstr[p].o1>=0 ) { // H_{+1,b}
	for ( int k=4*p; k<4*p+4; ++k){
	  if ( vrtx[k]>=0 ) {
	    Stck.push(vrtx[k]);
	    vrtx[k] = cmark;
	  } //end of 'if'
	} //end of 'for'
      } else { // H_{-1,i} <--> H_{-2,i}
	opstr[p].o1 = (opstr[p].o1==-1) ? -2 : -1 ;
      } //end of 'if opstr[p].o1>=0'
    } //end of 'while (ns>0)'

  } // end of member function 'QAQMC::flip_cluster'



  void QAQMC::init_conf(void) {

    if ( init==0 )
      random_conf();
    else 
      read_conf();
    
  } // end of member function 'QAQMC::init_conf'


  void QAQMC::random_conf()
  {
    for (int i=0; i<N; ++i) 
      spn2[i] = spn1[i] = ( ran() > 0.5e0 ? 1 : -1);

    for (int p=0; p<MM; ++p) {
      opstr[p].o1 = -1; 
      opstr[p].o2 = int( ran() * N );
    }
    
  } // random_conf



  void QAQMC::read_conf() {
    int k=0;
    std::ifstream conf;

    conf.open("conf.dat");

    if ( conf.is_open() ) {
      for (int i=0; i<N; ++i){
	conf >> spn1[i] >> spn2[i];
	++k;
      }

      for (int p=0; p<MM; ++p) {
	conf >> opstr[p].o1 >> opstr[p].o2;
	++k;
      }
    } else {
      std::cout << "Can't open file conf.dat \n";
      exit (1);
    } //end of 'if conf.is_open()'
    conf.close();

    if ( k!=N+MM ) {
      std::cout << " error in read conf! \n";
      exit (1);
    }

  }// end of member function 'QAQMC::read_conf'



  void QAQMC::write_conf() const {
    std::ofstream conf;

    conf.open("conf.dat",std::ios::trunc);

    if ( conf.is_open() ) {
      for (int i=0; i<N; ++i)
	conf << std::setw(5) << spn1[i] << std::setw(5) << spn2[i] << std::endl;
      
      for (int p=0; p<MM; ++p) 
	conf << std::setw(5) << opstr[p].o1 << std::setw(5) << opstr[p].o2 << std::endl;

    } else {
      std::cout << "Can't open file conf.dat";
    } //end of 'if conf.is_open()'
    conf.close();

  }// end of member function 'QAQMC::write_conf'



  void QAQMC::initialize(void) {

    // 1) Initialize RNG
    initran();

    // 2) read lattice parameters
    read_input();

    // 3) set parameters
    set_parameters();

    assert( tau >= npnt && tau%npnt == 0 );

    // 4) allocate arrays
    allocate_arrays();

    // 5) Initial configuration
    init_conf();

    // 6) construct lattice
    lattice();

    // 7) store the measurement points
    generate_points();
        
  } // end of member function 'QAQMC::initialize'


  void QAQMC::set_parameters()
  {
    N = int(pow(L,dd));  
    nb = N * dd;
    MM *= (2*N);
    tau = MM/2;

    Si=0.0e0;
    Sf=1.0e0; 
    r = 1.0;
    vel = Sf/pow(double(tau),r);
    
  } // set_parameters



  void QAQMC::allocate_arrays(void) {

    spn1 = new short signed int[N];
    spn2 = new short signed int[N];

    bst = new int *[2];
    for (int i=0; i<2; ++i) bst[i] = new int [nb];

    vrtx = new int [4*MM];
    frst = new int [N];
    last = new int [N];
    opstr = new optype [MM];

    et = new int [npnt];
    ef = new int [npnt];
    ei = new int [npnt];
    ma = new double [npnt];
    m2 = new double [npnt];
    m4 = new double [npnt];
    S_table = new double [npnt];

  } // end of member function 'QAQMC::allocate_arrays'



  void QAQMC::quench_S(int p) {
    S = Sf - vel * pow(fabs(tau-p), r);
  } // end of member function quench_S

  void QAQMC::generate_points(void) 
  {
    grid_size = tau/npnt;

    int n = 0;
    S = Si;
    for (int p = 0; p < MM/2; ++p) {
      if ( p < tau ) 
	quench_S(p+1);
      else if (p > tau )
	quench_S(p);
      
      if ( (n < npnt) && ((p+1)%grid_size == 0) ) 
	{
	  S_table[n] = S;
	  p_to_n[p] = n;
	  n_to_p[n] = p;
	  ++n;
	}
    }
    assert( n == npnt );
    assert( p_to_n.find(MM/2-1) != p_to_n.end() );

    /*
    S = Si;
    for (int p = 0; p < MM; ++p) {
      if (p < MM/2)
	quench_S(p+1);
      else if (p > MM/2)
	quench_S(p);
      std::cout << " p : " << p << ", S: " << S << std::endl;
    } //p

    std::cout << "--------------------------\n";
    for (int p = 0; p < MM/2; ++p)
      if ( p_to_n.find(p) != p_to_n.end() )
	std::cout << "p: " << p 
		  << ", n: " << p_to_n[p] 
		  << ", S: " << S_table[ p_to_n[p] ]
		  << std::endl;
    std::cout << "--------------------------\n";
    for (int p = MM/2; p < MM; ++p)
      if ( p_to_n.find(MM-2-p) != p_to_n.end() )
	std::cout << "p: " << p 
		  << ", n: " << p_to_n[MM-2-p] 
		  << ", S: " << S_table[ p_to_n[MM-2-p] ]
		  << std::endl;
    exit(0);
    */
    
  } // end of member function 'generate_points'



  void QAQMC::lattice() {
    //    The site index is : 0~N-1
    //    The bond index is : 0~nb-1
    //

    int i, l2; 
    int is, x1, x2, y1, y2, z1, z2;

    if ( dd == 1 ){

      for ( i=0; i<nb; i++ ) {
	bst[0][i]=i;
	bst[1][i]=(i+1)%L;
      }//end of 'for i=0...'

    } else if ( dd == 2 ) {

      for ( i=0; i<N; i++ ) {
	bst[0][i] = i;
	bst[0][i+N] = i;
	x1=i%L ; y1=i/L;
	bst[1][i] = (x1+1)%L + y1*L ;
	bst[1][i+N] = x1 + ((y1+1)%L) * L ;
      }//end of 'for i=0...'

    } else if ( dd == 3 ) {

      is=0; l2 = L*L;
      for( z1=0; z1<L; z1++ ){
	for( y1=0; y1<L; y1++ ){
	  for( x1=0; x1<L; x1++ ){

	    x2 = (1+x1)%L; // +x direction
	    y2 = y1;
	    z2 = z1;
	    bst[0][is] = is;
	    bst[1][is] = x2 + y2*L + z2*l2;

	    x2 = x1; // +y direction
	    y2 = (1+y1)%L;
	    z2 = z1;
	    bst[0][is+N] = is;
	    bst[1][is+N] = x2 + y2*L + z2*l2;

	    x2 = x1; // +z direction
	    y2 = y1;
	    z2 = (1+z2)%L;
	    bst[0][is+N*2] = is;
	    bst[1][is+N*2] = x2 + y2*L + z2*l2;

	    is++;
	  }//end of x1
	}//end of y1
      }//end of z1

    } else {
      std::cout << "1D ,2D, or 3D only! " << std::endl;
      exit (1);
    }// end if dd
    
  }// end of member function 'QAQMC::lattice'



  void QAQMC::read_input(void) {
    std::ifstream inputfile("input.in");

    if ( inputfile.is_open() ) {
      while ( !inputfile.eof() ) {
	inputfile >> dd >> L; 
	inputfile >> MM >> npnt;
	inputfile >> init >> istp >> mstp >> nbins;
      }
    } else {
      std::cout << "can not open input.in" << std::endl;
      exit (1);
    }    
    
  } //end of member function 'QAQMC::read_input'  



  void QAQMC::initran(void) {
    std::ifstream seedfile ("seed.in");

    if ( seedfile.is_open() ) {
      while ( ! seedfile.eof() ){
	for ( int i=0; i<4; ++i ) seedfile >> seed[i];
      }
    } else {
      std::cout << "Unable to open the seed file" << std::endl;
      exit (1);
    } //end of 'if seedfile.is_open()'
    seedfile.close();

    iir=1+int(fabs(seed[0]));
    jjr=1+int(fabs(seed[1]));
    kkr=1+int(fabs(seed[2]));
    nnr=seed[3];
  }// end of member function 'QAQMC::initran'



  double QAQMC::ran(void) {
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
  }// end of member function 'QAQMC::ran'



  QAQMC::~QAQMC() {

    delete [] spn1;
    delete [] spn2;
    
    for (int i=0; i<2; i++) delete [] bst[i];
    delete [] bst;

    delete [] vrtx;
    delete [] frst;
    delete [] last;
    delete [] opstr;

    delete [] et;
    delete [] ef;
    delete [] ei;
    delete [] ma;
    delete [] m2;
    delete [] m4;
    
    delete [] S_table;
  }


  void QAQMC::simulation_parameters(int& istp1, int& mstp1, int& nbins1) const
  {
    istp1 = istp; mstp1 = mstp; nbins1 = nbins;
  } 


}// end of namespace 'cwliu_QAQMC_namespace'
