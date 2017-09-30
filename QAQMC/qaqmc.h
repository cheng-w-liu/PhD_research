// FILE: qaqmc.h
//  header file for the class 'QAQMC'

#ifndef __CWLIU_QAQMC_CLASS
#define __CWLIU_QAQMC_CLASS

#include <stack>
#include <tr1/unordered_map> // provides unordered_map<Item1, Item2>

namespace cwliu_QAQMC_namespace {

  struct optype { int o1, o2; }; 

  class QAQMC 
  { 

  private:

    // variables for Random Numer Generator
    unsigned int seed[4];
    int iir, jjr, kkr, nnr;
    
    // Monte Carlo iteration parameters
    int init, istp, mstp, nbins;

    int npnt;       // no. of measurement points
    int grid_size;  // the interval between two measurement points
    int dd;         // dimension
    int L;          // size
    int N;          // no. of sites
    int nb;         // no. of bonds
    int MM;         // string length
    int tau;
    int cmark;      // cluster mark

    double p0;      // heat bath prob. of updating to H_{-1,i}=h (constant operator)
    double S;       // adiabatic evolution parameter S
    double Si;      // initial value of S
    double Sf;      // final value of S
    double r;
    double vel;

    signed short int *spn1; // |spin1> 
    signed short int *spn2; // |spin2> 

    int **bst;      // lattice structure         
    int *vrtx;      // linked vertices
    int *frst;      // the first appearance in the linked vertices of each site 
    int *last;      // the last appearance in the linked position of each site

    std::stack<int> Stck;

    optype *opstr;   // operator string   
    double *S_table; // store the s-value at each measurement point
    std::tr1::unordered_map<int,int> p_to_n;
    std::tr1::unordered_map<int,int> n_to_p;

    // Measurements
    int *et;      // E/N
    int *ef;      // field term
    int *ei;      // Ising term
    double *ma;   // |m|
    double *m2;   // m**2
    double *m4;   // m**4

    // ========== PRIVATE FUNCTIONs ==========//
    void initran(void);
    double ran(void);
    void read_input(void);
    void generate_points(void);
    void set_parameters(void);
    void allocate_arrays(void);
    void lattice(void);
    void init_conf(void);
    void random_conf(void);
    void read_conf(void);

    void quench_S(int p);    
    void diagonal_update(void);
    void heatbath(void);
    void vertices_link(void);
    void cluster_update(void);
    void visit_cluster(void);
    void flip_cluster(void);

  public:

    ~QAQMC();
    void initialize(void);
    void simulation_parameters(int& istp1, int& mstp1, int& nbins1) const;
    void mc_sweep(void);
    void write_conf(void) const;
    void write_data(void);
    void clean(void);
    void measure(void);

  };// end of class 'QAQMC'

}// end of namespace 'cwliu_QAQMC_namespace'

#endif
