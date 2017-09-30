// FILE: ising3d.c

#include <string>
#include <stdio.h>
#include <math.h>
#include <stdlib.h> // Provides rand(), RAND_MAX
#include <assert.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

using std::string;

#define BLOCKLx 4
#define BLOCKLy 4
#define BLOCKLz 4
#define	MyBit 1ULL
#define	N64bit 64

typedef unsigned long long int bit64;
typedef bit64 spin_t;

int L, D, N;
int init, istp, mstp, nbins;
double T;
double enrg, ma, m2, m4;

spin_t *spins;
int **nnbors;
float prob[13];

texture<float, 1, cudaReadModeElementType> Boltz;

__global__ void init_rand(int L, unsigned long long int seed, curandState_t *states) {

	   int global_x = blockIdx.x * blockDim.x + threadIdx.x;
	   int global_y = blockIdx.y * blockDim.y + threadIdx.y;
	   int global_z = blockIdx.z * blockDim.z + threadIdx.z;
	   int global_n = global_z * L * L + global_y * L + global_x;

           curand_init(seed, global_n, global_n, &states[global_n]);	   
	   
	   __syncthreads();
} // init_rand


__device__ float ran(curandState* global_state, int global_n) {
	   curandState_t local_state = global_state[global_n];
	   float r = curand_uniform(&local_state);
	   global_state[global_n] = local_state;
	   return r;
} // ran



__global__ void display_dims() {

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.y == 0) {
            printf("gDim.x : %i \n", gridDim.x);
            printf("bDim.x : %i \n", blockDim.x);
        }
    } 
    __syncthreads();
} // display_dims



__global__ void mc_updates(int L, curandState* global_state, spin_t *d_spins, int offset) {
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

	   __syncthreads();

	   local_x = threadIdx.x + 1;
	   local_y = threadIdx.y + 1;
	   local_z = threadIdx.z + 1;
	   local_n = local_z * (BLOCKLx+2)*(BLOCKLy+2) + local_y * (BLOCKLx+2) + local_x;
	   sj = local_spins[local_n];
	       
	   if ( (threadIdx.x + threadIdx.y + threadIdx.z + offset)%2 == 0 ) {

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
		   if ( ran(global_state, global_n) < tex1Dfetch(Boltz,dE+6) )
		       sj ^= mask;
	       } // b	       
               local_spins[local_n] = sj;

	       d_spins[global_n] = local_spins[local_n];
	       
	   } // end of "if (Idx.x + Idx.y + Idx.z + offset)%2 == 0"

	   __syncthreads();

} // mc_updates


// ========================================================================== //
void initialize();
void read_file();
void set_parameters();
void allocate_arrays();
void lattice();
void configuration();
void random_conf();
void read_conf();
void write_conf();
void probability();
void deallocate_arrays();
void clean();
void measure();
void write_data();
// ========================================================================== //


int main(int argc, char* argv[]) {
    curandState_t *devStates;
    spin_t *dev_spins;
    float *dev_prob;

    srand(time(NULL)); 
    initialize();

    dim3 block(BLOCKLx, BLOCKLy, BLOCKLz);
    dim3 grid(L/BLOCKLx, L/BLOCKLy, L/BLOCKLz);

    //display_dims<<<grid, block>>>();

    cudaMalloc((void **)&dev_spins, N*sizeof(spin_t));
    cudaMalloc((void **)&devStates, N*sizeof(curandState_t));     

    init_rand<<<grid, block>>>(L, rand(), devStates);

    cudaMalloc((void **) &dev_prob, (4*D+1) * sizeof(float));
    cudaMemcpy(dev_prob, prob, (4*D+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture(NULL, Boltz, dev_prob, (4*D+1)*sizeof(float));
        
    cudaMemcpy(dev_spins, spins, N*sizeof(spin_t), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < istp; ++i) {
    	mc_updates<<<grid, block>>>(L, devStates, dev_spins, 0);
	mc_updates<<<grid, block>>>(L, devStates, dev_spins, 1);
    }
    cudaMemcpy(spins, dev_spins, N*sizeof(spin_t), cudaMemcpyDeviceToHost);
    write_conf();

    for (int k = 0; k < nbins; ++k) {
        clean();
    	for (int i = 0; i < mstp; ++i) {
    	    mc_updates<<<grid, block>>>(L, devStates, dev_spins, 0);
	    mc_updates<<<grid, block>>>(L, devStates, dev_spins, 1);
            cudaMemcpy(spins, dev_spins, N*sizeof(spin_t), cudaMemcpyDeviceToHost);
	    measure();
	} // i-mstp
        write_data();
        write_conf();
    } // k-bin
    
    if (dev_prob != NULL) { cudaFree(dev_prob); dev_prob = NULL; }
    if (devStates != NULL) { cudaFree(devStates); devStates = NULL; }
    if (dev_spins != NULL) { cudaFree(dev_spins); dev_spins = NULL; }

    deallocate_arrays();
   
    return 0;
} // main



void write_data()
{
    FILE *ofptr;
    double dmstp = (double) mstp;

    enrg /= dmstp;
    ma /= dmstp;
    m2 /= dmstp;
    m4 /= dmstp;

    ofptr = fopen("data.dat","a");
    fprintf(ofptr,"%12.8f  %12.8f  %12.8f  %12.8f \n", enrg, ma, m2, m4);
    fclose(ofptr);

} // write_data



void clean()
{
    enrg = ma = m2 = m4 = 0.0e0;
} // clean


void measure() {
    int E = 0, j, b;
    int m[N64bit];
    bit64 mask, ss1, ss3, ss5;
    double dN = (double) N, dm , local_ma, local_m2, local_m4, d64 = (double) N64bit;
     
    for (b = 0; b < N64bit; ++b) m[b] = 0;

    for (j = 0; j < N; ++j) {
        ss1 = spins[j] ^ spins[ nnbors[j][1] ];
	ss3 = spins[j] ^ spins[ nnbors[j][3] ];
	ss5 = spins[j] ^ spins[ nnbors[j][5] ];
	for (b = 0; b < N64bit; ++b) {
            mask = (MyBit << b);
	    m[b] += ( (spins[j] & mask) ? 1 : -1 ); 
	    E += ( (ss1 & mask) ? -1 : 1 );
	    E += ( (ss3 & mask) ? -1 : 1 );
	    E += ( (ss5 & mask) ? -1 : 1 );
        } // b
    } // j

    enrg += (-(double) E)/(dN * (double) N64bit);
        
    local_ma = local_m2 = local_m4 = 0.0;
    for (b = 0; b < N64bit; ++b) {
        dm = (double) m[b]/dN;
        local_ma += fabs(dm);
        local_m2 += pow(dm,2.0);
        local_m4 += pow(dm,4.0);
    } //b                                                                                                         
    ma += local_ma/d64;
    m2 += local_m2/d64;
    m4 += local_m4/d64;

} // measure



void initialize()
{
    // 1) read-in input parameters
    read_file();

    // 2) set simulation parameters
    set_parameters();

    // 3) allocate arrays
    allocate_arrays();

    // 4) generate the 2D lattice
    lattice();   

    // 5) generate the initial configuration
    configuration();

    // 6) construct the probbility table
    probability();
    
} // initialize



void probability()
{
    float beta = 1.0e0/T;

    // e <--> 2 \sigma^B_i XOR \sigma^B_j - 1 
    for (int e = -6; e <= 6; ++e)
       	prob[e+6] = exp(2.0 * beta * (float) e);

} // probability



void configuration()
{
    if (init == 0)
        random_conf();
    else
        read_conf();

} // configuration



void random_conf()
{
    for (int i = 0; i < N; ++i) {
        spins[i] = 0;
        for(int b = 0; b < N64bit; ++b) {
            if (((double)rand())/((double)RAND_MAX) > 0.5)
                spins[i] ^= (MyBit << b);
        } // b 
    } // i
    
} // random_conf



void read_conf()
{
    FILE *fptr;
    fptr = fopen("spins.dat", "rt");
    if (fptr == NULL) { printf("can not open spins.dat"); exit(0); }
    for (int i = 0; i < N; ++i)
        fscanf(fptr, "%llu", &spins[i]);
    fclose(fptr);

} // read_conf



void write_conf()
{
  FILE *ofptr;
  int i;
  ofptr = fopen("spins.dat","w");
  for (i = 0; i < N; ++i)
    fprintf(ofptr,"%llu\n",spins[i]);
  fclose(ofptr);

} // write_conf



void lattice()
{
    int L2 = L * L;

    for (int z0 = 0; z0 < L; ++z0) {
        for (int y0 = 0; y0 < L; ++y0) {
            for (int x0 = 0; x0 < L; ++x0) {
                int x1 = (x0-1+L)%L;
	        int x2 = (x0+1)%L;
	        int y1 = (y0-1+L)%L;
	        int y2 = (y0+1)%L;
		int z1 = (z0-1+L)%L;
		int z2 = (z0+1)%L;

	        int j = z0 * L2 + y0 * L + x0;

	        nnbors[j][0] = z0 * L2 + y0 * L + x1;
	        nnbors[j][1] = z0 * L2 + y0 * L + x2;
	        nnbors[j][2] = z0 * L2 + y1 * L + x0;
	        nnbors[j][3] = z0 * L2 + y2 * L + x0;
	        nnbors[j][4] = z1 * L2 + y0 * L + x0;
	        nnbors[j][5] = z2 * L2 + y0 * L + x0;

            } // x0
        } // y0
    } // z0

} // lattice



void deallocate_arrays()
{
    if (spins != NULL) { free(spins); spins = NULL; }

    for (int i = 0; i < N; ++i) 
        if (nnbors[i] != NULL) 
            free(nnbors[i]); 
    if (nnbors != NULL) 
        free(nnbors);
    nnbors = NULL;

} // deallocate_arrays



void allocate_arrays()
{
    spins = (spin_t *) malloc(N * sizeof(spin_t));

    nnbors = (int **) malloc(N*sizeof(int*));
    for (int i = 0; i < N; ++i)
        nnbors[i] = (int *) malloc(2*D*sizeof(int));
    
} // allocate_arrays



void set_parameters()
{
    N = (int) pow(L, D);

} // set_parameters



void read_file()
{
    FILE *fptr;
    fptr = fopen("input.in", "rt");
    if (fptr == NULL) { 
        printf("can not open input.in"); 
	exit(0); 
    }
    fscanf(fptr,"%i %i %lf", &D, &L, &T);
    fscanf(fptr,"%i %i %i %i", &init, &istp, &mstp, &nbins);
    fclose(fptr);

} // read_file
