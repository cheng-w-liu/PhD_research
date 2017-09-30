// FILE: ising3d_q.c
//
// 1) H = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j , J > 0 for FM
//           
// 
//   2. Lattice labelings :
//
//
//                      j3 
//                      .     j4 (+z)
//                      |    .                                +z
//                      |   /                                /
//                      |  /                                /
//                      | /                                /
//                      |/                                /
//      j1 . ---------- . ---------- . j0 (+x)           .---------- +x
//                     /|                                |
//                    / |                                |
//                   /  |                                |
//                  /   |                                |
//                 .    |                                |
//               j5     .                                +y
//                      j2
//                     (+y)
//

#include <string>
#include <stdio.h>
#include <math.h>
#include <stdlib.h> // Provides rand(), RAND_MAX
#include <assert.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

using std::string;

#define D 3
#define BLOCKLx 4
#define BLOCKLy 4
#define BLOCKLz 4
#define	MyBit 1ULL
#define	N64bit 64

typedef unsigned long long int bit64;
typedef bit64 spin_t;
typedef double v_type;

int L, N;
int init, istp, mstp, nbins;
int tau, qpnt, tot_pnt, interval;
v_type T, Ti, Tf, vel, r;

__device__ __constant__ v_type Boltz[4*D+1];



__global__ void init_rand(int L, unsigned long long int seed, curandState_t *states) 
{

  int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  int global_z = blockIdx.z * blockDim.z + threadIdx.z;
  int global_n = global_z * L * L + global_y * L + global_x;

  curand_init(seed, global_n, global_n, &states[global_n]);	   
	   
  __syncthreads();

} // init_rand



__device__ v_type ran(curandState* global_state, int global_n) 
{
  curandState_t local_state = global_state[global_n];
  v_type rn = curand_uniform(&local_state);
  global_state[global_n] = local_state;
  return rn;
} // ran



__global__ void display_dims() 
{
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.y == 0) {
      printf("gDim.x : %i \n", gridDim.x);
      printf("bDim.x : %i \n", blockDim.x);
    }
  } 
  __syncthreads();

} // display_dims



__global__ void mc_updates(int L, curandState* global_state, spin_t *d_spins, int off) 
{
  __shared__ spin_t local_spins[(BLOCKLx+2)*(BLOCKLy+2)*(BLOCKLz+2)];
  int global_x, global_y, global_z, global_n;
  int nn_global_x, nn_global_y, nn_global_z, nn_global_n;
  int local_x, local_y, local_z, local_n;
  spin_t sj, ss0, ss1, ss2, ss3, ss4, ss5, mask;
  int xm, xp, ym, yp, zm, zp, L2 = L*L, b, dE;
	      
  global_x = blockIdx.x * blockDim.x + threadIdx.x;
  global_y = blockIdx.y * blockDim.y + threadIdx.y;
  global_z = blockIdx.z * blockDim.z + threadIdx.z;
  global_n = global_z * L2 + global_y * L + global_x;
	   
  local_x = threadIdx.x + 1;
  local_y = threadIdx.y + 1;
  local_z = threadIdx.z + 1;
  local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;

  local_spins[local_n] = d_spins[global_n];

  if (threadIdx.x == 0) {
    nn_global_x = ((blockIdx.x-1+gridDim.x)%gridDim.x)*BLOCKLx + BLOCKLx-1;
    nn_global_y = global_y;
    nn_global_z = global_z;
    nn_global_n = nn_global_z * L2 + nn_global_y * L + nn_global_x;
    local_x = 0;
    local_y = threadIdx.y + 1;
    local_z = threadIdx.z + 1;
    local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
    local_spins[local_n] = d_spins[nn_global_n];
  }

  if (threadIdx.x == BLOCKLx-1) {
    nn_global_x = ((blockIdx.x+1)%gridDim.x)*BLOCKLx;
    nn_global_y = global_y;
    nn_global_z = global_z;
    nn_global_n = nn_global_z * L2 + nn_global_y * L + nn_global_x;
    local_x = BLOCKLx+1;
    local_y = threadIdx.y + 1;
    local_z = threadIdx.z + 1;
    local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
    local_spins[local_n] = d_spins[nn_global_n];
  }

  if (threadIdx.y == 0) {
    nn_global_x = global_x;
    nn_global_y = ((blockIdx.y-1+gridDim.y)%gridDim.y)*BLOCKLy + BLOCKLy-1;
    nn_global_z = global_z;
    nn_global_n = nn_global_z * L2 + nn_global_y * L + nn_global_x;
    local_x = threadIdx.x + 1;
    local_y = 0;
    local_z = threadIdx.z + 1;
    local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
    local_spins[local_n] = d_spins[nn_global_n];	   
  }

  if (threadIdx.y == BLOCKLy-1) {
    nn_global_x = global_x;
    nn_global_y = ((blockIdx.y+1)%gridDim.y)*BLOCKLy;
    nn_global_z = global_z;
    nn_global_n = nn_global_z * L2 + nn_global_y * L + nn_global_x;
    local_x = threadIdx.x + 1;
    local_y = BLOCKLy+1;
    local_z = threadIdx.z + 1;
    local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
    local_spins[local_n] = d_spins[nn_global_n];	   
  }	   	   

  if (threadIdx.z == 0) {
    nn_global_x = global_x;
    nn_global_y = global_y;
    nn_global_z = ((blockIdx.z-1+gridDim.z)%gridDim.z)*BLOCKLz + BLOCKLz-1;
    nn_global_n = nn_global_z * L2 + nn_global_y * L + nn_global_x;
    local_x = threadIdx.x + 1;
    local_y = threadIdx.y + 1;
    local_z = 0;
    local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
    local_spins[local_n] = d_spins[nn_global_n];	   
  }	   	   

  if (threadIdx.z == BLOCKLz-1) {
    nn_global_x = global_x;
    nn_global_y = global_y;
    nn_global_z = ((blockIdx.z+1)%gridDim.z)*BLOCKLz;
    nn_global_n = nn_global_z * L2 + nn_global_y * L + nn_global_x;
    local_x = threadIdx.x + 1;
    local_y = threadIdx.y + 1;
    local_z = BLOCKLz+1;
    local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
    local_spins[local_n] = d_spins[nn_global_n];	   
  }	   	   

  //__syncthreads();

  local_x = threadIdx.x + 1;
  local_y = threadIdx.y + 1;
  local_z = threadIdx.z + 1;
  local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
  sj = local_spins[local_n];

  __syncthreads();
	       
  if ( (threadIdx.x + threadIdx.y + threadIdx.z + off)%2 == 0 ) {

    xm = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x-1;
    xp = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x+1;
    ym = local_z * (BLOCKLx+2)*(BLOCKLy+2) + (local_y-1) * (BLOCKLx+2) + local_x;
    yp = local_z * (BLOCKLx+2)*(BLOCKLy+2) + (local_y+1) * (BLOCKLx+2) + local_x;
    zm = (local_z-1) * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
    zp = (local_z+1) * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;

    ss0 = sj ^ local_spins[xm];
    ss1 = sj ^ local_spins[xp];
    ss2 = sj ^ local_spins[ym];
    ss3 = sj ^ local_spins[yp];
    ss4 = sj ^ local_spins[zm];
    ss5 = sj ^ local_spins[zp];

    for (b = 0; b < N64bit; ++b) {
      dE = 0;
      // dE <--> 2 \sigma^B_i XOR \sigma^B_j - 1 
      mask = (MyBit << b);
      dE += (ss0 & mask) ? 1 : -1;
      dE += (ss1 & mask) ? 1 : -1;
      dE += (ss2 & mask) ? 1 : -1;
      dE += (ss3 & mask) ? 1 : -1;
      dE += (ss4 & mask) ? 1 : -1;
      dE += (ss5 & mask) ? 1 : -1;
      if ( ran(global_state, global_n) < Boltz[dE+6] ) {
	sj ^= mask;
      }
    } // b	       
    local_spins[local_n] = sj;

    d_spins[global_n] = local_spins[local_n];
	       
  } // end of "if (Idx.x + Idx.y + Idx.z + off)%2 == 0"

  __syncthreads();

} // mc_updates


// ========================================================================== //
void initialize();
void read_file();
void set_parameters();
void lattice(int *nnbors);
void configuration(spin_t *spins);
void random_conf(spin_t *spins);
void read_conf(spin_t *spins);
void write_conf(spin_t *spins);
void checkout_configuration(spin_t *spins, spin_t *saved_spins);
void save_configuration(spin_t *spins, spin_t *saved_spins);
void initial_temperature();
void quench_temperature(int t);
void tT_tables(int *t_table, v_type *T_table);
void probability(v_type *h_prob);
void clean(double *enrg, double *maga, double *mag2, double *mag4);
void measure(spin_t *spins,int *nnbors,int n,double *enrg,double *maga,double *mag2,double *mag4);
void write_data(int *t_table,v_type *T_table,double *enrg,double *maga,double *mag2,double *mag4);
void check_stop();
// ========================================================================== //


int main(int argc, char* argv[]) {
    curandState_t *devStates;
    spin_t *spins, *saved_spins, *dev_spins;
    v_type *host_prob;
    v_type *T_table;
    int *t_table;
    int *nnbors;
    double *enrg;
    double *maga;
    double *mag2;
    double *mag4;
    int i, j, k, t, n, steps = 10;

    srand(time(NULL)); 

    initialize();

    dim3 block(BLOCKLx, BLOCKLy, BLOCKLz);
    assert(L > 0);
    dim3 grid(L/BLOCKLx, L/BLOCKLy, L/BLOCKLz);

    nnbors = (int *) malloc(N*2*D * sizeof(int));
    lattice(nnbors);

    t_table = (int *) malloc(tot_pnt * sizeof(int));
    T_table = (v_type *) malloc(tot_pnt * sizeof(v_type));
    tT_tables(t_table, T_table);

    spins = (spin_t *) malloc( N * sizeof(spin_t) ); assert(spins != NULL);
    saved_spins = (spin_t *) malloc( N * sizeof(spin_t) ); assert(saved_spins != NULL);
    host_prob = (v_type *) malloc((4*D+1) * sizeof(v_type));    
    configuration(spins);

    cudaMalloc((void **)&dev_spins, N*sizeof(spin_t)); assert(dev_spins != NULL);
    cudaMalloc((void **)&devStates, N*sizeof(curandState_t));     

    init_rand<<<grid, block>>>(L, rand(), devStates);
    cudaDeviceSynchronize();

    enrg = (double *) malloc(tot_pnt * sizeof(double));
    maga = (double *) malloc(tot_pnt * sizeof(double));
    mag2 = (double *) malloc(tot_pnt * sizeof(double));
    mag4 = (double *) malloc(tot_pnt * sizeof(double));

    initial_temperature(); probability(host_prob);
    cudaMemcpyToSymbol(Boltz,host_prob,(4*D+1)*sizeof(v_type),0,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_spins, spins, N*sizeof(spin_t), cudaMemcpyHostToDevice);    
    for (i = 0; i < istp; ++i) {
    	mc_updates<<<grid, block>>>(L, devStates, dev_spins, 0);
	cudaDeviceSynchronize();
	mc_updates<<<grid, block>>>(L, devStates, dev_spins, 1);
	cudaDeviceSynchronize();
    } // i
    cudaMemcpy(spins, dev_spins, N*sizeof(spin_t), cudaMemcpyDeviceToHost);
    save_configuration(spins, saved_spins);
    write_conf(spins);

    for (k = 0; k < nbins; ++k) {
      clean(enrg, maga, mag2, mag4);
      for (i = 0; i < mstp; ++i) {
	initial_temperature(); probability(host_prob);
	cudaMemcpyToSymbol(Boltz,host_prob,(4*D+1)*sizeof(v_type),0,cudaMemcpyHostToDevice);

	checkout_configuration(spins, saved_spins);
	cudaMemcpy(dev_spins, spins, N*sizeof(spin_t), cudaMemcpyHostToDevice);
	for(j = 0; j < steps; ++j) {
	  mc_updates<<<grid, block>>>(L, devStates, dev_spins, 0);
	  cudaDeviceSynchronize();
	  mc_updates<<<grid, block>>>(L, devStates, dev_spins, 1);
	  cudaDeviceSynchronize();
	} // i
	cudaMemcpy(spins, dev_spins, N*sizeof(spin_t), cudaMemcpyDeviceToHost);
	save_configuration(spins, saved_spins);	    

	n = 0;
	for (t = 0; t <= tau; ++t) {
	  quench_temperature(t); probability(host_prob);
	  cudaMemcpyToSymbol(Boltz,host_prob,(4*D+1)*sizeof(v_type),0,cudaMemcpyHostToDevice);
	  mc_updates<<<grid, block>>>(L, devStates, dev_spins, 0);
	  cudaDeviceSynchronize();
	  mc_updates<<<grid, block>>>(L, devStates, dev_spins, 1);
	  cudaDeviceSynchronize();
	  if (t % interval == 0) {
	    cudaMemcpy(spins, dev_spins, N*sizeof(spin_t), cudaMemcpyDeviceToHost);
	    measure(spins, nnbors, n++, enrg, maga, mag2, mag4);
	  } // if
	} // t
      } // i-mstp
      write_data(t_table, T_table, enrg, maga, mag2, mag4);
      write_conf(spins);
      check_stop();
    } // k-bin
    
    if (devStates != NULL) { cudaFree(devStates); devStates = NULL; }
    if (dev_spins != NULL) { cudaFree(dev_spins); dev_spins = NULL; }

    if (nnbors != NULL) { free(nnbors); nnbors = NULL; }
    if (t_table != NULL) { free(t_table); t_table = NULL; }
    if (T_table != NULL) { free(T_table); T_table = NULL; }
    if (spins != NULL) { free(spins); spins = NULL; }
    if (saved_spins != NULL) { free(saved_spins); saved_spins = NULL; }
    if (host_prob != NULL) { free(host_prob); host_prob = NULL; }

    if (enrg != NULL) { free(enrg); enrg = NULL; }
    if (maga != NULL) { free(maga); maga = NULL; }
    if (mag2 != NULL) { free(mag2); mag2 = NULL; }
    if (mag4 != NULL) { free(mag4); mag4 = NULL; }

    return 0;
} // main



void write_data(int *t_table,v_type *T_table,double *enrg,double *maga,double *mag2,double *mag4)
{
    int n;
    FILE *ofptr;
    double dmstp = (double) mstp;

    ofptr = fopen("data.dat","a");
    for(n = 0; n < tot_pnt; ++n) {
        enrg[n] /= dmstp;
	maga[n] /= dmstp;
	mag2[n] /= dmstp;
	mag4[n] /= dmstp;
        fprintf(ofptr,"%8i %12.8f %14.8f %12.8f %12.8f %12.8f\n", 
                t_table[n], T_table[n], enrg[n], maga[n], mag2[n], mag4[n]);
    } // b
    fclose(ofptr);

} // write_data



void clean(double *enrg, double *maga, double *mag2, double *mag4)
{
    int n;
    for (n = 0; n < tot_pnt; ++n) {
        enrg[n] = maga[n] = mag2[n] = mag4[n] = 0.0;
    }

} // clean



void measure(spin_t *spins,int *nnbors,int n,double *enrg,double *maga,double *mag2,double *mag4)
{
  int E = 0, j, b;
  int m[N64bit];
  bit64 mask, ss0, ss2, ss4;
  double dN = (double) N, dm , local_ma, local_m2, local_m4, d64 = (double) N64bit;
     
  for (b = 0; b < N64bit; ++b) m[b] = 0;

  for (j = 0; j < N; ++j) {
    ss0 = spins[j] ^ spins[ nnbors[j*2*D+0] ];
    ss2 = spins[j] ^ spins[ nnbors[j*2*D+2] ];
    ss4 = spins[j] ^ spins[ nnbors[j*2*D+4] ];
    for (b = 0; b < N64bit; ++b) {
      mask = (MyBit << b);
      m[b] += ( (spins[j] & mask) ? 1 : -1 ); 

      // dE <--> 2 \sigma^B_i XOR \sigma^B_j - 1 
      E += ( (ss0 & mask) ? 1 : -1 );
      E += ( (ss2 & mask) ? 1 : -1 );
      E += ( (ss4 & mask) ? 1 : -1 );
    } // b
  } // j

  enrg[n] += ((double) E)/(dN * d64);
        
  local_ma = local_m2 = local_m4 = 0.0;
  for (b = 0; b < N64bit; ++b) {
    dm = (double) m[b]/dN;
    local_ma += fabs(dm);
    local_m2 += pow(dm,2.0);
    local_m4 += pow(dm,4.0);
  } //b                                                                                                         
  maga[n] += local_ma/d64;
  mag2[n] += local_m2/d64;
  mag4[n] += local_m4/d64;

} // measure



void initial_temperature() { T = Ti; }



void quench_temperature(int t) 
{
    T = Tf + vel * pow((v_type)(tau-t), r);
} //



void initialize()
{
    // 1) read-in input parameters
    read_file();

    // 2) set simulation parameters
    set_parameters();
    
} // initialize



void probability(v_type *h_prob)
{
    v_type beta = 1.0e0/T;

    // e <--> 2 \sigma^B_i XOR \sigma^B_j - 1 
    for (int e = -6; e <= 6; ++e) {
       	h_prob[e+6] = exp(2.0 * beta * (v_type) e);
    }

} // probability



void checkout_configuration(spin_t *spins, spin_t *saved_spins)
{
    int i;
    for (i = 0; i < N; ++i) {
        spins[i] = saved_spins[i];
    }

} // checkout_configuration



void save_configuration(spin_t *spins, spin_t *saved_spins)
{
    int i;
    for (i = 0; i < N; ++i) {
        saved_spins[i] = spins[i];
    }

} // save_configuration



void configuration(spin_t *spins)
{
    if (init == 0) {
        random_conf(spins);
    } else {
        read_conf(spins);
    }

} // configuration



void random_conf(spin_t *spins)
{
    for (int i = 0; i < N; ++i) {
        spins[i] = 0;
        for(int b = 0; b < N64bit; ++b) {
            if (((double)rand())/((double)RAND_MAX) > 0.5) {
                spins[i] ^= (MyBit << b);
            } // if
        } // b 
    } // i
    
} // random_conf



void read_conf(spin_t *spins)
{
    FILE *fptr;
    fptr = fopen("spins.dat", "rt");
    if (fptr == NULL) { printf("can not open spins.dat"); exit(0); }
    for (int i = 0; i < N; ++i) {
        fscanf(fptr, "%llu", &spins[i]);
    }
    fclose(fptr);

} // read_conf



void write_conf(spin_t *spins)
{
  FILE *ofptr;
  int i;
  ofptr = fopen("spins.dat","w");
  for (i = 0; i < N; ++i) {
    fprintf(ofptr,"%llu\n",spins[i]);
  }
  fclose(ofptr);

} // write_conf



void lattice(int *nnbors)
{
    int L2 = L * L;

    for (int z0 = 0; z0 < L; ++z0) {
        for (int y0 = 0; y0 < L; ++y0) {
            for (int x0 = 0; x0 < L; ++x0) {
	        int x1 = (x0+1)%L;
                int x2 = (x0-1+L)%L;
	        int y1 = (y0+1)%L;
	        int y2 = (y0-1+L)%L;
		int z1 = (z0+1)%L;
		int z2 = (z0-1+L)%L;

	        int j = z0 * L2 + y0 * L + x0;

	        nnbors[j*2*D+0] = z0 * L2 + y0 * L + x1;
	        nnbors[j*2*D+1] = z0 * L2 + y0 * L + x2;
	        nnbors[j*2*D+2] = z0 * L2 + y1 * L + x0;
	        nnbors[j*2*D+3] = z0 * L2 + y2 * L + x0;
	        nnbors[j*2*D+4] = z1 * L2 + y0 * L + x0;
	        nnbors[j*2*D+5] = z2 * L2 + y0 * L + x0;

            } // x0
        } // y0
    } // z0

} // lattice



void tT_tables(int *t_table, v_type *T_table)
{
    int t, n = 0;

    initial_temperature();
    for (t = 0; t <= tau; ++t) {
        quench_temperature(t);
	if (t % interval == 0) {
	    t_table[n] = t;
	    T_table[n] = T;
	    ++n;
	}
    } // t
    assert(n == tot_pnt);

} // tT_tables



void set_parameters()
{
    v_type Tc = 1.0/0.22169;

    N = L * L * L;

    Ti = 1.5 * Tc;
    Tf = 0.9 * Tc; // 1.0
    r = 1.0;
    vel = (Ti-Tf)/pow((v_type) tau, r);
    
    tot_pnt = qpnt + 1;

    interval = tau/qpnt;

} // set_parameters



void read_file()
{
    FILE *fptr;
    fptr = fopen("input.in", "rt");
    if (fptr == NULL) { 
        printf("can not open input.in"); 
	exit(0); 
    }
    fscanf(fptr,"%i", &L);
    fscanf(fptr,"%i %i", &tau, &qpnt);
    fscanf(fptr,"%i %i %i %i", &init, &istp, &mstp, &nbins);
    fclose(fptr);

    assert( tau%qpnt == 0 );

} // read_file


void check_stop()
{
    FILE *fptr;
    int i;

    fptr = fopen("stop.txt", "rt");
    if (fptr == NULL) { 
        printf("can not open stop.txt"); 
	return; 
    }
    fscanf(fptr,"%i", &i);
    fclose(fptr);

    if (i != 0) {
        fptr = fopen("stop.txt","a");
	fprintf(fptr,"%s \n", "stopped");	
	fclose(fptr);
	exit(0);
    } // if
} // check_stop

