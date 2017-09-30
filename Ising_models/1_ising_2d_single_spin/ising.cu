// FILE: ising2d.c

#include <string>
#include <stdio.h>
#include <math.h>
#include <stdlib.h> // Provides rand(), RAND_MAX
#include <assert.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

using std::string;

#define BLOCKLx 8
#define BLOCKLy 8

int L, D, N;
int init, istp, mstp, nbins;
double T;
double enrg, ma, m2, m4;

int *spins;
int **nnbors;
float prob[9];

texture<float, 1, cudaReadModeElementType> Boltz;

__global__ void init_rand(int L, unsigned long long int seed, curandState_t *states) {

	   int global_x = blockIdx.x * blockDim.x + threadIdx.x;
	   int global_y = blockIdx.y * blockDim.y + threadIdx.y;
	   int global_n = global_y * L + global_x;

           curand_init(seed, global_n, global_n, &states[global_n]);	   
	   
	   __syncthreads();
} // init_rand


__device__ float ran(curandState* global_state, int global_n) {
	   curandState_t local_state = global_state[global_n];
	   float r = curand_uniform(&local_state);
	   global_state[global_n] = local_state;
	   return r;
} // ran


__global__ void mc_updates(int L, curandState* global_state, int *d_spins, int offset) {
	   __shared__ int local_spins[(BLOCKLx+2)*(BLOCKLy+2)];
	   int global_x, global_y, global_n;
	   int nn_global_x, nn_global_y, nn_global_n;
	   int local_x, local_y, local_n;
	   int e;

	   //assert(gridDim.x * blockDim.x == L);
	      
	   global_x = blockIdx.x * blockDim.x + threadIdx.x;
	   global_y = blockIdx.y * blockDim.y + threadIdx.y;
	   global_n = global_y * L + global_x;
	   
	   local_x = threadIdx.x + 1;
	   local_y = threadIdx.y + 1;
	   local_n = local_y * (BLOCKLx+2) + local_x;
	   local_spins[local_n] = d_spins[global_n];

	   if (threadIdx.x == 0) {
	      nn_global_x = ((blockIdx.x-1+gridDim.x)%gridDim.x)*BLOCKLx + BLOCKLx-1;
	      nn_global_y = global_y;
	      nn_global_n = nn_global_y * L + nn_global_x;
	      local_x = 0;
	      local_y = threadIdx.y + 1;
	      local_n = local_y * (BLOCKLx+2) + local_x;
	      local_spins[local_n] = d_spins[nn_global_n];
	   }

	   if (threadIdx.x == BLOCKLx-1) {
	      nn_global_x = ((blockIdx.x+1)%gridDim.x)*BLOCKLx;
	      nn_global_y = global_y;
	      nn_global_n = nn_global_y * L + nn_global_x;
	      local_x = BLOCKLx+1;
	      local_y = threadIdx.y + 1;
	      local_n = local_y * (BLOCKLx+2) + local_x;
	      local_spins[local_n] = d_spins[nn_global_n];
	   }

	   if (threadIdx.y == 0) {
	      nn_global_x = global_x;
	      nn_global_y = ((blockIdx.y-1+gridDim.y)%gridDim.y)*BLOCKLy + BLOCKLy-1;
	      nn_global_n = nn_global_y * L + nn_global_x;
	      local_x = threadIdx.x + 1;
	      local_y = 0;
	      local_n = local_y * (BLOCKLx+2) + local_x;
	      local_spins[local_n] = d_spins[nn_global_n];	   
	   }

	   if (threadIdx.y == BLOCKLy-1) {
	      nn_global_x = global_x;
	      nn_global_y = ((blockIdx.y+1)%gridDim.y)*BLOCKLy;
	      nn_global_n = nn_global_y * L + nn_global_x;
	      local_x = threadIdx.x + 1;
	      local_y = BLOCKLy+1;
	      local_n = local_y * (BLOCKLx+2) + local_x;
	      local_spins[local_n] = d_spins[nn_global_n];	   
	   }	   	   

	   __syncthreads();

	   local_x = threadIdx.x + 1;
	   local_y = threadIdx.y + 1;
	   local_n = local_y * (BLOCKLx+2) + local_x;
	       
	   if ( (threadIdx.x + threadIdx.y + offset)%2 == 0 ) {

	       int lf = local_y * (BLOCKLx+2) + local_x - 1;
	       int rt = local_y * (BLOCKLx+2) + local_x + 1;
	       int up = (local_y - 1) * (BLOCKLx+2) + local_x;
	       int dw = (local_y + 1) * (BLOCKLx+2) + local_x;
	       
	       e=local_spins[local_n]*(local_spins[lf]+local_spins[rt]+local_spins[up]+local_spins[dw]);

	       if ( ran(global_state, global_n) < tex1Dfetch(Boltz,e+4) ) 
	           local_spins[local_n] *= -1;

   	       d_spins[global_n] = local_spins[local_n];	

	   } // end of "if (Idx.x + Idx.y + offset)%2 == 0"

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
    int *dev_spins;
    float *dev_prob;

    srand(time(NULL)); 
    initialize();

    dim3 block(BLOCKLx, BLOCKLy);
    dim3 grid(L/BLOCKLx, L/BLOCKLy);    

    cudaMalloc((void **)&dev_spins, N*sizeof(int));
    cudaMalloc((void **)&devStates, N*sizeof(curandState_t));     

    init_rand<<<grid, block>>>(L, 762198, devStates);

    cudaMalloc((void **) &dev_prob, (4*D+1) * sizeof(float));
    cudaMemcpy(dev_prob, prob, (4*D+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture(NULL, Boltz, dev_prob, (4*D+1)*sizeof(float));
        
    cudaMemcpy(dev_spins, spins, N*sizeof(int), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < istp; ++i) {
    	mc_updates<<<grid, block>>>(L, devStates, dev_spins, 0);
	mc_updates<<<grid, block>>>(L, devStates, dev_spins, 1);
    }
    cudaMemcpy(spins, dev_spins, N*sizeof(int), cudaMemcpyDeviceToHost);
    write_conf();

    for (int k = 0; k < nbins; ++k) {
        clean();
    	for (int i = 0; i < mstp; ++i) {
    	    mc_updates<<<grid, block>>>(L, devStates, dev_spins, 0);
	    mc_updates<<<grid, block>>>(L, devStates, dev_spins, 1);
            cudaMemcpy(spins, dev_spins, N*sizeof(int), cudaMemcpyDeviceToHost);
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
    int E = 0, M = 0;
    double m;

    for (int j = 0; j < N; ++j) {
        E += spins[j] * ( spins[ nnbors[j][0] ] + spins[ nnbors[j][3] ] );
        M += spins[j];
    } // j

    m = ((double) M)/((double) N);

    enrg += (-(double) E)/((double) N);
    ma += fabs(m);
    m2 += pow(m,2);
    m4 += pow(m,4);

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

    for (int e = -4; e <= 4; ++e)
       	prob[e+4] = exp(-2.0 * beta * (float) e);

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
    for (int i = 0; i < N; ++i) 
    	spins[i] = ((double) rand()/(double) RAND_MAX) > 0.5 ? 1 : -1;
    
} // random_conf



void read_conf()
{
    FILE *fptr;
    fptr = fopen("spins.dat", "rt");
    if (fptr == NULL) { printf("can not open spins.dat"); exit(0); }
    for (int i = 0; i < N; ++i)
        fscanf(fptr, "%d", &spins[i]);
    fclose(fptr);

} // read_conf



void write_conf()
{
  FILE *ofptr;
  int i;
  ofptr = fopen("spins.dat","w");
  for (i = 0; i < N; ++i)
    fprintf(ofptr,"%i\n",spins[i]);
  fclose(ofptr);

} // write_conf



void lattice()
{
    for (int y0 = 0; y0 < L; ++y0) {
        for (int x0 = 0; x0 < L; ++x0) {
            int x1 = (x0+1)%L;
	    int x2 = (x0-1+L)%L;
	    int y1 = (y0-1+L)%L;
	    int y2 = (y0+1)%L;
	    int j = y0 * L + x0;
	    nnbors[j][0] = y0 * L + x1;
	    nnbors[j][1] = y0 * L + x2;
	    nnbors[j][2] = y1 * L + x0;
	    nnbors[j][3] = y2 * L + x0;
        } // x0
    } // y0

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
    spins = (int *) malloc(N * sizeof(int));

    nnbors = (int **) malloc(N*sizeof(int*));
    for (int i = 0; i < N; ++i)
        nnbors[i] = (int *) malloc(N*sizeof(int));
    
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
