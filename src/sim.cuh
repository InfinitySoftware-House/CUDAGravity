#pragma once
#include <cstdint>

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
    int   preset   = 1;        // initial-condition template (1..6)
};

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

    SimParams p_;
    int   nbodies_   = 0;
    int   capacity_  = 0;      // total tree-node slots (2n)
    float worldRadius_ = 1.2f;
    bool  treeBuilt_ = false;  // root COM valid once a tree has been built

    // Per-body arrays (size n).
    float* posx_ = nullptr; float* posy_ = nullptr; float* posz_ = nullptr;
    float* velx_ = nullptr; float* vely_ = nullptr; float* velz_ = nullptr;
    float* accx_ = nullptr; float* accy_ = nullptr; float* accz_ = nullptr;
    float* mass_ = nullptr;

    // LBVH tree-node arrays. Node id space [0,2n-1): internal [0,n-1), leaves [n-1,2n-1).
    float* nm_   = nullptr;                                  // node mass
    float* ncx_  = nullptr; float* ncy_ = nullptr; float* ncz_ = nullptr;   // node COM
    float* nminx_ = nullptr; float* nminy_ = nullptr; float* nminz_ = nullptr;  // node AABB
    float* nmaxx_ = nullptr; float* nmaxy_ = nullptr; float* nmaxz_ = nullptr;
    int*   left_   = nullptr;   // internal-node children (size n)
    int*   right_  = nullptr;
    int*   parent_ = nullptr;   // size 2n
    int*   visited_ = nullptr;  // bottom-up arrival counters (size n)
    unsigned* code_ = nullptr;  // Morton keys (size n)
    int*   idx_     = nullptr;  // sorted body indices (size n)
};
