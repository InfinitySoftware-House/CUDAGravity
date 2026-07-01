#pragma once
#include <cstdint>
#include <cmath>
#include <vector_types.h>   // float4 (lightweight, no full CUDA runtime needed)

// Public host-side interface to the 3D CUDA Barnes-Hut simulation.
// Implementation lives in barnes_hut.cu (compiled by nvcc).

struct SimParams {
    int   n        = 100000;   // number of stars
    int   nGas     = 0;        // gas tracer particles (rendered as nebulae)
    int   nDust    = 0;        // dust tracer particles (rendered as dust lanes)
    float theta    = 0.5f;     // Barnes-Hut opening angle
    float dt       = 0.0008f;  // timestep
    float eps      = 0.0040f;  // softening length
    float G        = 1.0f;     // gravitational constant
    float damping  = 1.0f;     // velocity damping per step (1.0 = none)
    uint32_t seed  = 1234u;
    int   preset   = 1;        // initial-condition template (1..7)
    bool  dtGiven  = false;    // user passed --dt (presets then keep their hands off it)

    // --- accuracy / physics toggles ---
    bool  useQuadrupole = true;  // quadrupole (2nd-order) Barnes-Hut multipoles
    bool  haloOn        = true;  // static analytic NFW dark-matter halo per galaxy
    float haloMassFrac  = 8.0f;  // M_halo / M_baryonic (typical DM:baryon ~ 5-10)
    float haloRsFrac    = 3.0f;  // NFW scale radius, in units of the disk scale length Rd
    float omegaM        = 0.30f; // matter density parameter (preset 7 cosmology)
    float omegaL        = 0.70f; // dark-energy density parameter (flat universe: omegaM+omegaL=1)
};

// Physical unit system (only an interpretation layer -- the solver itself is
// unitless with G=1). Chosen so that a 1.0-mass, 1.0-radius disk galaxy comes
// out at a realistic scale: 1 length unit = 10 kpc, 1 mass unit = 1e11 Msun,
// velocity/time follow from G=1. This is the standard N-body ("Henon") unit
// convention used throughout galactic dynamics (Binney & Tremaine).
namespace units {
    constexpr double kLengthKpc   = 10.0;                         // 1 sim length unit
    constexpr double kMassMsun    = 1.0e11;                       // 1 sim mass unit
    constexpr double kG_astro     = 4.30091e-6;                   // kpc*(km/s)^2/Msun
    // v_unit = sqrt(G_astro * M_unit / L_unit)  [km/s]
    inline double velocityKms() { return std::sqrt(kG_astro * kMassMsun / kLengthKpc); }
    // t_unit = L_unit / v_unit, converted kpc/(km/s) -> Myr
    inline double timeMyr() {
        const double kpcInKm = 3.0857e16;
        const double secPerMyr = 3.1557e13;
        return (kLengthKpc * kpcInKm / velocityKms()) / secPerMyr;
    }
}

class Simulation {
public:
    explicit Simulation(const SimParams& p);
    ~Simulation();

    // Advance the simulation by one timestep.
    void step();

    // Copy current body positions + speed into an interleaved device buffer
    // (stride 4 floats: x, y, z, speed). Used to feed an OpenGL VBO.
    void copyToRenderBuffer(float* d_dst);

    // Mass-weighted centre of mass of all bodies. Read from the tree root, so
    // it is valid after the first step()/computeAccelOnly(); falls back to the
    // origin before the tree has been built.
    void centerOfMass(float& x, float& y, float& z);

    // Diagnostics / test entry points.
    void computeAccelOnly();                       // BH accel for current state
    void computeDirectReference(float* d_ax, float* d_ay, float* d_az);
    void accelError(const float* d_refx, const float* d_refy, const float* d_refz,
                    double& meanRelErr, double& maxRelErr);
    double totalEnergy();                           // O(N^2) — small N only
    void dumpPositions(const char* path);           // text "x y z" per body

    // Body layout is contiguous: [0,nStars) stars, then gas, then dust.
    // Gas and dust are low-mass tracer bodies — they are advected by the full
    // gravitational field but barely perturb it, so they shear into realistic
    // spiral lanes / tidal structure on their own.
    int   n()           const { return nbodies_; }   // total bodies
    int   nStars()      const { return p_.n; }
    int   nGas()        const { return p_.nGas; }
    int   nDust()       const { return p_.nDust; }
    float worldRadius() const { return worldRadius_; }

private:
    void buildTree();
    void summarize();
    void computeForces();       // tree traversal only (monopole [+ quadrupole])
    void applyExternalForces(); // NFW halo(s) + cosmological Lambda term, added on top

    SimParams p_;
    int   nbodies_   = 0;
    int   capacity_  = 0;      // total tree-node slots (2n)
    float worldRadius_ = 1.2f;
    bool  treeBuilt_ = false;  // root COM valid once a tree has been built
    float lambdaTerm_ = 0.0f;  // cosmological-constant accel coefficient (preset 7 only)

    // Per-body arrays (size n).
    float* posx_ = nullptr; float* posy_ = nullptr; float* posz_ = nullptr;
    float* velx_ = nullptr; float* vely_ = nullptr; float* velz_ = nullptr;
    float* accx_ = nullptr; float* accy_ = nullptr; float* accz_ = nullptr;
    float* mass_ = nullptr;

    // LBVH tree-node arrays. Node id space [0,2n-1): internal [0,n-1), leaves [n-1,2n-1).
    float4* ncom_ = nullptr;    // node centre of mass + mass packed (x,y,z,m)
    float*  nsize2_ = nullptr;  // node size^2, precomputed for the BH opening test
    float* nminx_ = nullptr; float* nminy_ = nullptr; float* nminz_ = nullptr;  // node AABB
    float* nmaxx_ = nullptr; float* nmaxy_ = nullptr; float* nmaxz_ = nullptr;
    // Traceless quadrupole moment tensor per node (Qxx,Qxy,Qxz,Qyy packed + Qyz;
    // Qzz = -(Qxx+Qyy) is derived on the fly). Only allocated when p_.useQuadrupole.
    float4* nquadA_ = nullptr;  // (Qxx,Qxy,Qxz,Qyy)
    float*  nquadB_ = nullptr;  // Qyz
    int*   left_   = nullptr;   // internal-node children (size n)
    int*   right_  = nullptr;
    int*   parent_ = nullptr;   // size 2n
    int*   visited_ = nullptr;  // bottom-up arrival counters (size n)
    unsigned* code_ = nullptr;  // Morton keys (size n)
    int*   idx_     = nullptr;  // sorted body indices (size n)
};
