// CUDA Barnes-Hut N-body simulation (3D).
//
// Tree: a lock-free LBVH (linear bounding volume hierarchy).  This is a
// deadlock-free way to build a spatial tree on the GPU and works as a proper
// Barnes-Hut acceleration structure: far-away subtrees are approximated by
// their centre of mass when their size/distance ratio is below `theta`.
//
// Pipeline per step:
//   1. bounds     -> enclosing cube of all bodies
//   2. morton     -> 30-bit Z-order key per body
//   3. sort       -> bodies ordered along the space-filling curve (thrust)
//   4. radix tree -> Karras 2012 binary tree over the sorted keys
//   5. summarize  -> bottom-up centre of mass + AABB (atomic, lock-free)
//   6. forces     -> per-body tree traversal with the Barnes-Hut criterion
//   7. integrate  -> symplectic Euler (kick-drift)
//
// Node id space (size 2n-1):  internal nodes [0, n-1),  leaves [n-1, 2n-1).
// Leaf node (n-1)+p holds sorted body position p -> real body index idx[p].

#include "sim.cuh"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CUDA_CHECK(x) do {                                            \
    cudaError_t err__ = (x);                                          \
    if (err__ != cudaSuccess) {                                       \
        std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",        \
            #x, __FILE__, __LINE__, cudaGetErrorString(err__));       \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

static const int BLOCK = 256;
static inline int grid(int nthreads) { return (nthreads + BLOCK - 1) / BLOCK; }

// ----- device globals -----------------------------------------------------
__device__ float d_minx, d_maxx, d_miny, d_maxy, d_minz, d_maxz;
__device__ float d_cubeMinX, d_cubeMinY, d_cubeMinZ, d_cubeInv;

// Static analytic dark-matter halos (NFW) + cosmological Lambda term. Tiny
// fixed-size table in constant memory -- broadcast to every thread for
// free, no extra global-memory traffic per body.
#define MAX_HALOS 4
__constant__ int   d_haloCount;
__constant__ float d_haloX[MAX_HALOS], d_haloY[MAX_HALOS], d_haloZ[MAX_HALOS];
__constant__ float d_haloMs[MAX_HALOS], d_haloRs[MAX_HALOS];
__constant__ float d_lambdaTerm;

// ----- atomic float min/max ----------------------------------------------
__device__ __forceinline__ float atomicMinf(float* addr, float val) {
    int* ai = (int*)addr; int old = *ai, assumed;
    do { assumed = old;
         if (__int_as_float(assumed) <= val) break;
         old = atomicCAS(ai, assumed, __float_as_int(val));
    } while (old != assumed);
    return __int_as_float(old);
}
__device__ __forceinline__ float atomicMaxf(float* addr, float val) {
    int* ai = (int*)addr; int old = *ai, assumed;
    do { assumed = old;
         if (__int_as_float(assumed) >= val) break;
         old = atomicCAS(ai, assumed, __float_as_int(val));
    } while (old != assumed);
    return __int_as_float(old);
}

// ----- bounding box -------------------------------------------------------
__global__ void initBoundsKernel() {
    d_minx = d_miny = d_minz =  1e30f;
    d_maxx = d_maxy = d_maxz = -1e30f;
}
__global__ void boundsKernel(const float* px, const float* py, const float* pz, int n) {
    __shared__ float smnx, smxx, smny, smxy, smnz, smxz;
    if (threadIdx.x == 0) { smnx=smny=smnz= 1e30f; smxx=smxy=smxz=-1e30f; }
    __syncthreads();
    float mnx=1e30f,mxx=-1e30f,mny=1e30f,mxy=-1e30f,mnz=1e30f,mxz=-1e30f;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        float x=px[i],y=py[i],z=pz[i];
        mnx=fminf(mnx,x); mxx=fmaxf(mxx,x);
        mny=fminf(mny,y); mxy=fmaxf(mxy,y);
        mnz=fminf(mnz,z); mxz=fmaxf(mxz,z);
    }
    atomicMinf(&smnx,mnx); atomicMaxf(&smxx,mxx);
    atomicMinf(&smny,mny); atomicMaxf(&smxy,mxy);
    atomicMinf(&smnz,mnz); atomicMaxf(&smxz,mxz);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMinf(&d_minx,smnx); atomicMaxf(&d_maxx,smxx);
        atomicMinf(&d_miny,smny); atomicMaxf(&d_maxy,smxy);
        atomicMinf(&d_minz,smnz); atomicMaxf(&d_maxz,smxz);
    }
}
__global__ void setCubeKernel() {
    float cx=0.5f*(d_minx+d_maxx), cy=0.5f*(d_miny+d_maxy), cz=0.5f*(d_minz+d_maxz);
    float h = 0.5f*fmaxf(d_maxx-d_minx, fmaxf(d_maxy-d_miny, d_maxz-d_minz));
    h = h*1.0001f + 1e-6f;
    d_cubeMinX = cx-h; d_cubeMinY = cy-h; d_cubeMinZ = cz-h;
    d_cubeInv  = 1.0f/(2.0f*h);
}

// ----- Morton codes -------------------------------------------------------
__device__ __forceinline__ unsigned expandBits(unsigned v) { // 10-bit -> spread
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
__global__ void mortonKernel(const float* px, const float* py, const float* pz,
                             unsigned* code, int* idx, int n) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        float x = (px[i]-d_cubeMinX)*d_cubeInv;
        float y = (py[i]-d_cubeMinY)*d_cubeInv;
        float z = (pz[i]-d_cubeMinZ)*d_cubeInv;
        unsigned xi = min(max((unsigned)(x*1024.0f),0u),1023u);
        unsigned yi = min(max((unsigned)(y*1024.0f),0u),1023u);
        unsigned zi = min(max((unsigned)(z*1024.0f),0u),1023u);
        code[i] = (expandBits(xi)<<2) | (expandBits(yi)<<1) | expandBits(zi);
        idx[i]  = i;
    }
}

// ----- Karras radix tree --------------------------------------------------
__device__ __forceinline__ int delta(const unsigned* code, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    unsigned a = code[i], b = code[j];
    if (a == b) return 32 + __clz((unsigned)(i ^ j));  // tie-break with index
    return __clz(a ^ b);
}
__global__ void buildTreeKernel(const unsigned* code, int n,
                                int* left, int* right, int* parent) {
    if (blockIdx.x*blockDim.x+threadIdx.x == 0) parent[0] = -1;  // root
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n-1; i += blockDim.x*gridDim.x) {
        int d = (delta(code,n,i,i+1) - delta(code,n,i,i-1)) >= 0 ? 1 : -1;
        int dmin = delta(code,n,i,i-d);

        int lmax = 2;
        while (delta(code,n,i,i+lmax*d) > dmin) lmax <<= 1;
        int l = 0;
        for (int t = lmax>>1; t > 0; t >>= 1)
            if (delta(code,n,i,i+(l+t)*d) > dmin) l += t;
        int j = i + l*d;

        int dnode = delta(code,n,i,j);
        int s = 0;
        for (int div = 2; ; div <<= 1) {
            int t = (l + div - 1) / div;                // ceil(l/div)
            if (delta(code,n,i,i+(s+t)*d) > dnode) s += t;
            if (t == 1) break;
        }
        int gamma = i + s*d + min(d, 0);

        int lc = (min(i,j) == gamma)     ? (n-1+gamma)     : gamma;
        int rc = (max(i,j) == gamma + 1) ? (n-1+gamma+1)   : (gamma+1);
        left[i] = lc; right[i] = rc;
        parent[lc] = i; parent[rc] = i;
    }
}

// ----- bottom-up centre of mass + AABB (lock-free) ------------------------
__global__ void initLeavesKernel(const float* px, const float* py, const float* pz,
                                 const float* mass, const int* idx, int n,
                                 float4* ncom, float* nsize2,
                                 float* nminx, float* nminy, float* nminz,
                                 float* nmaxx, float* nmaxy, float* nmaxz,
                                 float4* nquadA, float* nquadB,
                                 int* visited) {
    for (int p = blockIdx.x*blockDim.x+threadIdx.x; p < n; p += blockDim.x*gridDim.x) {
        int node = (n-1) + p;
        int b = idx[p];
        float x=px[b], y=py[b], z=pz[b], m=mass[b];
        ncom[node]=make_float4(x,y,z,m);
        nsize2[node]=0.0f;               // leaf has no extent
        nminx[node]=nmaxx[node]=x;
        nminy[node]=nmaxy[node]=y;
        nminz[node]=nmaxz[node]=z;
        if (nquadA) { nquadA[node]=make_float4(0,0,0,0); nquadB[node]=0.0f; } // point mass -> zero quadrupole
        if (p < n-1) visited[p] = 0;     // reset internal-node counters
    }
}
__global__ void summarizeKernel(const int* parent, const int* left, const int* right,
                                int n, float4* ncom, float* nsize2,
                                float* nminx, float* nminy, float* nminz,
                                float* nmaxx, float* nmaxy, float* nmaxz,
                                float4* nquadA, float* nquadB,
                                int* visited) {
    for (int p = blockIdx.x*blockDim.x+threadIdx.x; p < n; p += blockDim.x*gridDim.x) {
        int cur = parent[(n-1) + p];
        __threadfence();
        while (cur != -1) {
            if (atomicAdd(&visited[cur], 1) == 0) break;   // first child waits
            int a = left[cur], b = right[cur];
            float4 ca = ncom[a], cb = ncom[b];
            float ma = ca.w, mb = cb.w, m = ma + mb;
            float inv = (m > 0.0f) ? 1.0f/m : 0.0f;
            float cx = (ma*ca.x + mb*cb.x) * inv;
            float cy = (ma*ca.y + mb*cb.y) * inv;
            float cz = (ma*ca.z + mb*cb.z) * inv;
            ncom[cur] = make_float4(cx, cy, cz, m);
            float mnx=fminf(nminx[a],nminx[b]), mny=fminf(nminy[a],nminy[b]), mnz=fminf(nminz[a],nminz[b]);
            float mxx=fmaxf(nmaxx[a],nmaxx[b]), mxy=fmaxf(nmaxy[a],nmaxy[b]), mxz=fmaxf(nmaxz[a],nmaxz[b]);
            nminx[cur]=mnx; nminy[cur]=mny; nminz[cur]=mnz;
            nmaxx[cur]=mxx; nmaxy[cur]=mxy; nmaxz[cur]=mxz;
            float s = fmaxf(mxx-mnx, fmaxf(mxy-mny, mxz-mnz));
            nsize2[cur] = s*s;             // precompute size^2 -> forces read 1 float
            if (nquadA) {
                // Exact parallel-axis combination of two traceless quadrupole
                // tensors about their common new centre of mass (Steiner's
                // theorem generalised to the quadrupole moment):
                //   Q_new = Qa + ma*(3 da(x)da - |da|^2 I)
                //         + Qb + mb*(3 db(x)db - |db|^2 I)
                // where da,db are the child COMs relative to the merged COM.
                float4 qa = nquadA[a]; float qayz = nquadB[a];
                float4 qb = nquadA[b]; float qbyz = nquadB[b];
                float dax=ca.x-cx, day=ca.y-cy, daz=ca.z-cz;
                float dbx=cb.x-cx, dby=cb.y-cy, dbz=cb.z-cz;
                float da2=dax*dax+day*day+daz*daz, db2=dbx*dbx+dby*dby+dbz*dbz;
                float qxx = qa.x + qb.x + ma*(3.0f*dax*dax-da2) + mb*(3.0f*dbx*dbx-db2);
                float qxy = qa.y + qb.y + ma*(3.0f*dax*day)     + mb*(3.0f*dbx*dby);
                float qxz = qa.z + qb.z + ma*(3.0f*dax*daz)     + mb*(3.0f*dbx*dbz);
                float qyy = qa.w + qb.w + ma*(3.0f*day*day-da2) + mb*(3.0f*dby*dby-db2);
                float qyz = qayz + qbyz + ma*(3.0f*day*daz)     + mb*(3.0f*dby*dbz);
                nquadA[cur] = make_float4(qxx,qxy,qxz,qyy);
                nquadB[cur] = qyz;
            }
            __threadfence();
            if (cur == 0) break;
            cur = parent[cur];
        }
    }
}

// ----- force calculation --------------------------------------------------
// Barnes-Hut acceptance criterion (nsize2 < theta^2 r^2) is unaffected by
// multipole order: it decides when the *approximation* is trusted, and the
// quadrupole term below just makes that approximation more accurate at the
// same theta (or lets theta grow for the same accuracy -> fewer nodes opened).
__global__ void forcesKernel(const float* px, const float* py, const float* pz,
                             const int* idx, const int* left, const int* right,
                             const float4* __restrict__ ncom, const float* __restrict__ nsize2,
                             const float4* __restrict__ nquadA, const float* __restrict__ nquadB,
                             float* ax, float* ay, float* az,
                             int n, float theta, float eps2, float G, bool useQuad) {
    const float theta2 = theta*theta;
    int stack[64];
    // Iterate bodies in Morton-sorted order so neighbouring threads in a warp
    // traverse nearly identical tree paths -> far less branch divergence.
    for (int p = blockIdx.x*blockDim.x+threadIdx.x; p < n; p += blockDim.x*gridDim.x) {
        const int i = idx[p];
        const float bx=px[i], by=py[i], bz=pz[i];
        float fx=0.0f, fy=0.0f, fz=0.0f;

        int sp = 0; stack[sp++] = 0;          // root internal node
        while (sp > 0) {
            int node = stack[--sp];
            // One packed load covers both COM (xyz) and mass (w) for any node.
            float4 c = ncom[node];
            float dx=c.x-bx, dy=c.y-by, dz=c.z-bz;
            float r2 = dx*dx+dy*dy+dz*dz+eps2;
            bool isLeaf = node >= n-1;
            if (isLeaf && idx[node-(n-1)] == i) continue;   // self
            if (isLeaf || nsize2[node] < theta2*r2) {        // leaf, or far enough -> approximate
                float inv = rsqrtf(r2);
                float inv2 = inv*inv, inv3 = inv2*inv;
                float f = G*c.w*inv3;
                fx+=f*dx; fy+=f*dy; fz+=f*dz;
                if (useQuad && !isLeaf) {
                    // Quadrupole correction to the point-mass (monopole) term
                    // (Barnes 1986 / Hernquist 1987 tree codes). Note dx,dy,dz
                    // here point from the field body TOWARD the source (matches
                    // the monopole term above), which is the *negative* of the
                    // "R from source to field point" convention the textbook
                    // formula a_k = G(Q_kj R_j)/R^5 - (5/2)G(Q_ij R_i R_j)R_k/R^7
                    // is written in; substituting R=-dx flips the overall sign.
                    float4 qa = nquadA[node]; float qyz = nquadB[node];
                    float qxx=qa.x, qxy=qa.y, qxz=qa.z, qyy=qa.w, qzz=-(qxx+qyy);
                    float inv5 = inv3*inv2, inv7 = inv5*inv2;
                    float Qrr = qxx*dx*dx + qyy*dy*dy + qzz*dz*dz
                              + 2.0f*qxy*dx*dy + 2.0f*qxz*dx*dz + 2.0f*qyz*dy*dz;
                    float coef = 2.5f*G*Qrr*inv7;
                    fx += -G*(qxx*dx+qxy*dy+qxz*dz)*inv5 + coef*dx;
                    fy += -G*(qxy*dx+qyy*dy+qyz*dz)*inv5 + coef*dy;
                    fz += -G*(qxz*dx+qyz*dy+qzz*dz)*inv5 + coef*dz;
                }
            } else if (sp < 62) {             // open
                stack[sp++] = left[node]; stack[sp++] = right[node];
            }
        }
        ax[i]=fx; ay[i]=fy; az[i]=fz;
    }
}

// External, non-tree accelerations layered on top of the N-body gravity:
// static analytic NFW dark-matter halo(s) (one per galaxy) and, for the
// cosmological preset, the Lambda (dark-energy) term of the Friedmann
// acceleration equation. Both are O(1) per body -- negligible next to the
// tree traversal -- so real dark-matter/cosmology physics costs almost
// nothing on top of the N-body solve, which is what keeps this practical on
// a single consumer GPU instead of needing a live N-body DM component.
__global__ void externalAccelKernel(const float* px, const float* py, const float* pz,
                                    float* ax, float* ay, float* az, int n, float G) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        float bx=px[i], by=py[i], bz=pz[i];
        float fx=0.0f, fy=0.0f, fz=0.0f;
        for (int h = 0; h < d_haloCount; ++h) {
            float dx=d_haloX[h]-bx, dy=d_haloY[h]-by, dz=d_haloZ[h]-bz;
            float r2 = dx*dx+dy*dy+dz*dz;
            float r  = sqrtf(r2) + 1e-6f;
            float xr = r/d_haloRs[h];
            float menc = d_haloMs[h] * (logf(1.0f+xr) - xr/(1.0f+xr));  // NFW enclosed mass
            float f = G*menc/(r2*r);
            fx+=f*dx; fy+=f*dy; fz+=f*dz;
        }
        if (d_lambdaTerm != 0.0f) { fx += d_lambdaTerm*bx; fy += d_lambdaTerm*by; fz += d_lambdaTerm*bz; }
        ax[i]+=fx; ay[i]+=fy; az[i]+=fz;
    }
}

// Leapfrog (kick-drift-kick, symplectic, 2nd-order) integrator. Splitting a
// single-force-eval step into two half-kicks around one drift lets us reuse
// the same acceleration for the end of this step and the start of the next,
// so it costs exactly one tree build + force pass per step -- same as the
// old 1st-order symplectic-Euler scheme -- but conserves energy/angular
// momentum far better over long integrations (essential for anything
// claiming to be a real orbit/structure-formation simulation).
__global__ void kickKernel(float* vx, float* vy, float* vz,
                           const float* ax, const float* ay, const float* az,
                           int n, float dtHalf, float dampHalf) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        vx[i]=(vx[i]+ax[i]*dtHalf)*dampHalf;
        vy[i]=(vy[i]+ay[i]*dtHalf)*dampHalf;
        vz[i]=(vz[i]+az[i]*dtHalf)*dampHalf;
    }
}
__global__ void driftKernel(float* px, float* py, float* pz,
                            const float* vx, const float* vy, const float* vz,
                            int n, float dt) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        px[i]+=vx[i]*dt; py[i]+=vy[i]*dt; pz[i]+=vz[i]*dt;
    }
}

__global__ void directKernel(const float* px, const float* py, const float* pz,
                             const float* mass, float* ax, float* ay, float* az,
                             int n, float eps2, float G) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        float bx=px[i],by=py[i],bz=pz[i],fx=0,fy=0,fz=0;
        for (int j=0;j<n;++j){ if(j==i)continue;
            float dx=px[j]-bx,dy=py[j]-by,dz=pz[j]-bz;
            float r2=dx*dx+dy*dy+dz*dz+eps2; float inv=rsqrtf(r2);
            float f=G*mass[j]*inv*inv*inv; fx+=f*dx; fy+=f*dy; fz+=f*dz; }
        ax[i]=fx; ay[i]=fy; az[i]=fz;
    }
}

__global__ void renderCopyKernel(const float* px, const float* py, const float* pz,
                                 const float* vx, const float* vy, const float* vz,
                                 float* dst, int n) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        float sp = sqrtf(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]);
        dst[i*4+0]=px[i]; dst[i*4+1]=py[i]; dst[i*4+2]=pz[i]; dst[i*4+3]=sp;
    }
}

// =========================================================================
//  Host class
// =========================================================================
Simulation::Simulation(const SimParams& p) : p_(p), nbodies_(p.n + p.nGas + p.nDust) {
    capacity_ = 2*nbodies_;                   // total tree nodes (2n-1, rounded)

    auto fb = sizeof(float)*nbodies_;
    auto fn = sizeof(float)*capacity_;
    CUDA_CHECK(cudaMalloc(&posx_, fb)); CUDA_CHECK(cudaMalloc(&posy_, fb)); CUDA_CHECK(cudaMalloc(&posz_, fb));
    CUDA_CHECK(cudaMalloc(&velx_, fb)); CUDA_CHECK(cudaMalloc(&vely_, fb)); CUDA_CHECK(cudaMalloc(&velz_, fb));
    CUDA_CHECK(cudaMalloc(&accx_, fb)); CUDA_CHECK(cudaMalloc(&accy_, fb)); CUDA_CHECK(cudaMalloc(&accz_, fb));
    CUDA_CHECK(cudaMalloc(&mass_, fb));
    // tree node arrays
    CUDA_CHECK(cudaMalloc(&ncom_, sizeof(float4)*capacity_));
    CUDA_CHECK(cudaMalloc(&nsize2_, fn));
    CUDA_CHECK(cudaMalloc(&nminx_,fn)); CUDA_CHECK(cudaMalloc(&nminy_,fn)); CUDA_CHECK(cudaMalloc(&nminz_,fn));
    CUDA_CHECK(cudaMalloc(&nmaxx_,fn)); CUDA_CHECK(cudaMalloc(&nmaxy_,fn)); CUDA_CHECK(cudaMalloc(&nmaxz_,fn));
    if (p.useQuadrupole) {
        CUDA_CHECK(cudaMalloc(&nquadA_, sizeof(float4)*capacity_));
        CUDA_CHECK(cudaMalloc(&nquadB_, fn));
    }
    CUDA_CHECK(cudaMalloc(&left_,  sizeof(int)*nbodies_));
    CUDA_CHECK(cudaMalloc(&right_, sizeof(int)*nbodies_));
    CUDA_CHECK(cudaMalloc(&parent_,sizeof(int)*capacity_));
    CUDA_CHECK(cudaMalloc(&visited_,sizeof(int)*nbodies_));
    CUDA_CHECK(cudaMalloc(&code_, sizeof(unsigned)*nbodies_));
    CUDA_CHECK(cudaMalloc(&idx_,  sizeof(int)*nbodies_));

    // -------- initial conditions (preset-driven, with a realistic IMF) --------
    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<float> hx(nbodies_),hy(nbodies_),hz(nbodies_),
                       hvx(nbodies_),hvy(nbodies_),hvz(nbodies_),hm(nbodies_);
    const float G=p.G, eps=p.eps;

    // Salpeter-like stellar IMF: dN/dm ~ m^-2.35 over [0.1, 50] solar masses.
    // Most stars are light; a few are very massive -> realistic clumpy gravity.
    auto imf = [&]() -> float {
        const float a=2.35f, mlo=0.1f, mhi=50.0f;
        float lo=powf(mlo,1.0f-a), hi=powf(mhi,1.0f-a);
        return powf(lo + uni(rng)*(hi-lo), 1.0f/(1.0f-a));
    };

    // Static analytic NFW dark-matter halo(s): one registered per galaxy disk
    // (see addDisk below), applied as an external background acceleration in
    // externalAccelKernel. Cheap (O(1)/body) way to give each galaxy a
    // realistic DM:baryon mass ratio and flat rotation curve without doubling
    // the N-body particle count.
    std::vector<float> haloX,haloY,haloZ,haloMs,haloRs;
    auto addHalo = [&](float cx,float cy,float cz,float Mgal,float Rd){
        if (!p.haloOn || (int)haloX.size() >= MAX_HALOS) return;
        float rs = Rd*p.haloRsFrac;
        float m1 = logf(2.0f)-0.5f;                 // NFW mass-profile m(x=1)
        float Ms = p.haloMassFrac*Mgal/m1;           // normalises Menc(rs) = haloMassFrac*Mgal
        haloX.push_back(cx); haloY.push_back(cy); haloZ.push_back(cz);
        haloMs.push_back(Ms); haloRs.push_back(rs);
    };
    // Halo's contribution to the enclosed mass at radius r, for circular-
    // velocity initial conditions -- must match addHalo's (Ms,rs) exactly, or
    // disks seeded with a live halo start out badly out of equilibrium and
    // collapse/fly apart in the first few dynamical times.
    auto haloMenc = [&](float r,float Mgal,float Rd) -> float {
        if (!p.haloOn) return 0.0f;
        float rs = Rd*p.haloRsFrac;
        float Ms = p.haloMassFrac*Mgal/(logf(2.0f)-0.5f);
        float xr = r/rs;
        return Ms*(logf(1.0f+xr) - xr/(1.0f+xr));
    };

    // Rotating disk galaxy into slice [s, s+cnt): centre, bulk velocity, mass,
    // radius, spin sign, and inclination (tilt about the x-axis).
    auto addDisk = [&](int s,int cnt,float cx,float cy,float cz,
                       float bvx,float bvy,float bvz,float Mgal,float Rd,
                       float spin,float inc){
        addHalo(cx,cy,cz,Mgal,Rd);
        float Mbulge=0.25f*Mgal, ci=cosf(inc), si=sinf(inc);
        double wsum=0;
        for(int k=0;k<cnt;++k){ int i=s+k;
            float u=uni(rng);
            float r=Rd*(0.02f+0.98f*u*u);             // surface density ~ 1/r
            float ang=uni(rng)*6.2831853f;
            float x=r*cosf(ang), y=r*sinf(ang);
            float th=Rd*0.05f*(uni(rng)-0.5f)*(uni(rng)-0.5f)*4.0f*expf(-r/Rd*1.5f);
            float Menc=Mbulge+Mgal*(r*r)/(Rd*Rd)+haloMenc(r,Mgal,Rd);
            float vc=spin*sqrtf(G*Menc/sqrtf(r*r+eps*eps));
            float jit=0.92f+0.10f*uni(rng);
            float vx=-vc*sinf(ang)*jit, vy=vc*cosf(ang)*jit, vz=0.0f;
            float y2=y*ci - th*si,  z2=y*si + th*ci;   // incline about x
            float vy2=vy*ci - vz*si, vz2=vy*si + vz*ci;
            hx[i]=cx+x; hy[i]=cy+y2; hz[i]=cz+z2;
            hvx[i]=bvx+vx; hvy[i]=bvy+vy2; hvz[i]=bvz+vz2;
            float w=imf(); hm[i]=w; wsum+=w;
        }
        float scale=(float)(Mgal/wsum);               // normalise stellar masses
        for(int k=0;k<cnt;++k) hm[s+k]*=scale;
        int nb=cnt/200+1;                             // heavy central bulge
        for(int k=0;k<nb;++k){ int i=s+k;
            hx[i]=cx+(uni(rng)-0.5f)*0.02f; hy[i]=cy+(uni(rng)-0.5f)*0.02f; hz[i]=cz+(uni(rng)-0.5f)*0.02f;
            hvx[i]=bvx; hvy[i]=bvy; hvz[i]=bvz; hm[i]=Mbulge/nb;
        }
    };

    // Uniform sphere into slice [s,s+cnt): vexp>0 expands (Hubble-like),
    // ~0 collapses; rot adds solid-body rotation about z.
    auto addSphere = [&](int s,int cnt,float cx,float cy,float cz,float R,
                         float Mtot,float vexp,float rot){
        double wsum=0;
        for(int k=0;k<cnt;++k){ int i=s+k;
            float rr=R*cbrtf(uni(rng));
            float ct=2.0f*uni(rng)-1.0f, st=sqrtf(fmaxf(0.0f,1.0f-ct*ct)), ph=uni(rng)*6.2831853f;
            float x=rr*st*cosf(ph), y=rr*st*sinf(ph), z=rr*ct;
            hx[i]=cx+x; hy[i]=cy+y; hz[i]=cz+z;
            hvx[i]=vexp*x - rot*y; hvy[i]=vexp*y + rot*x; hvz[i]=vexp*z;
            float w=imf(); hm[i]=w; wsum+=w;
        }
        float scale=(float)(Mtot/wsum);
        for(int k=0;k<cnt;++k) hm[s+k]*=scale;
    };

    // Low-mass tracer disk: same orbital kinematics as addDisk so the gas/dust
    // co-rotates and shears into spiral lanes, but negligible mass (it is pushed
    // around by the stars, it does not push back). thickK scales disk thickness
    // (gas puffy, dust razor-thin); rext scales radial extent.
    const float mTracer = 2.0e-7f;            // per-particle mass (<< star masses)
    // Low-mass tracer disk. armPhase!=... biases particles onto a 2-arm
    // logarithmic spiral (real gas/dust hugs the arms, it does not fill the
    // disk); armStr in [0,1] sets how tightly. thickK scales vertical thickness.
    auto addTracerDisk = [&](int s,int cnt,float cx,float cy,float cz,
                             float bvx,float bvy,float bvz,float Mgal,float Rd,
                             float spin,float inc,float thickK,float rext,
                             float armPhase,float armStr){
        float Mbulge=0.25f*Mgal, ci=cosf(inc), si=sinf(inc);
        for(int k=0;k<cnt;++k){ int i=s+k;
            // accept/reject toward the spiral arms (gas avoids inter-arm voids)
            float r, ang; int tries=0;
            do { float u=uni(rng); r=Rd*(0.06f+rext*u*u); ang=uni(rng)*6.2831853f;
                 float arm=0.5f+0.5f*sinf(2.0f*ang - 5.5f*logf(r+0.12f) + armPhase);
                 float pAcc=(1.0f-armStr) + armStr*arm*arm*arm;
                 if(uni(rng) < pAcc) break;
            } while(++tries<6);
            ang += 0.10f*(uni(rng)-0.5f);          // small scatter off the ridge
            float x=r*cosf(ang), y=r*sinf(ang);
            float th=Rd*thickK*(uni(rng)-0.5f)*(uni(rng)-0.5f)*4.0f*expf(-r/Rd*1.5f);
            float Menc=Mbulge+Mgal*(r*r)/(Rd*Rd)+haloMenc(r,Mgal,Rd);
            float vc=spin*sqrtf(G*Menc/sqrtf(r*r+eps*eps));
            float jit=0.97f+0.05f*uni(rng);
            float vx=-vc*sinf(ang)*jit, vy=vc*cosf(ang)*jit, vz=0.0f;
            float y2=y*ci - th*si, z2=y*si + th*ci;
            float vy2=vy*ci - vz*si, vz2=vy*si + vz*ci;
            hx[i]=cx+x; hy[i]=cy+y2; hz[i]=cz+z2;
            hvx[i]=bvx+vx; hvy[i]=bvy+vy2; hvz[i]=bvz+vz2;
            hm[i]=mTracer;
        }
    };
    auto addTracerSphere = [&](int s,int cnt,float cx,float cy,float cz,float R,
                               float vexp,float rot){
        for(int k=0;k<cnt;++k){ int i=s+k;
            float rr=R*cbrtf(uni(rng));
            float ct=2.0f*uni(rng)-1.0f, st=sqrtf(fmaxf(0.0f,1.0f-ct*ct)), ph=uni(rng)*6.2831853f;
            float x=rr*st*cosf(ph), y=rr*st*sinf(ph), z=rr*ct;
            hx[i]=cx+x; hy[i]=cy+y; hz[i]=cz+z;
            hvx[i]=vexp*x - rot*y; hvy[i]=vexp*y + rot*x; hvz[i]=vexp*z;
            hm[i]=mTracer;
        }
    };

    // Body ranges: [0,nS) stars | [gB,gB+nG) gas | [dB,dB+nD) dust.
    const int nS=p.n, nG=p.nGas, nD=p.nDust;
    const int gB=nS, dB=nS+nG;
    // Seed one preset component's gas+dust over [gB+go, ..) / [dB+do, ..).
    // Returns by advancing the running gas/dust cursors via lambda capture.
    int go=0, doff=0;
    auto gasDiskFrac = [&](float frac,float cx,float cy,float cz,
                           float bvx,float bvy,float bvz,float Mgal,float Rd,
                           float spin,float inc){
        int cg=(int)(nG*frac), cd=(int)(nD*frac);
        float armPhase=uni(rng)*6.2831853f;       // gas + dust share the arms
        // gas: moderately arm-biased, thin; dust: tightly arm-biased, razor-thin
        addTracerDisk(gB+go, cg, cx,cy,cz, bvx,bvy,bvz, Mgal,Rd, spin,inc, 0.035f,1.05f, armPhase,0.78f);
        addTracerDisk(dB+doff,cd, cx,cy,cz, bvx,bvy,bvz, Mgal,Rd, spin,inc, 0.012f,0.95f, armPhase,0.90f);
        go+=cg; doff+=cd;
    };
    auto gasSphereFrac = [&](float frac,float cx,float cy,float cz,float R,
                             float vexp,float rot){
        int cg=(int)(nG*frac), cd=(int)(nD*frac);
        addTracerSphere(gB+go, cg, cx,cy,cz, R*1.05f, vexp,rot);
        addTracerSphere(dB+doff,cd, cx,cy,cz, R*0.95f, vexp,rot);
        go+=cg; doff+=cd;
    };

    const int n=nS;                          // stars occupy [0,nS)
    switch(p.preset){
        default:
        case 1:  // single spiral galaxy
            addDisk(0,n, 0,0,0, 0,0,0, 1.0f,1.0f, +1.0f, 0.35f);
            gasDiskFrac(1.0f, 0,0,0, 0,0,0, 1.0f,1.0f, +1.0f, 0.35f);
            break;
        case 2:{ // two equal galaxies, grazing collision
            int h=n/2;
            addDisk(0,h,   -0.9f, 0.20f,0,  0.32f,-0.05f,0, 0.7f,0.6f, +1.0f,  0.40f);
            addDisk(h,n-h,  0.9f,-0.20f,0, -0.32f, 0.05f,0, 0.7f,0.6f, +1.0f, -0.70f);
            gasDiskFrac(0.5f, -0.9f, 0.20f,0,  0.32f,-0.05f,0, 0.7f,0.6f, +1.0f,  0.40f);
            gasDiskFrac(0.5f,  0.9f,-0.20f,0, -0.32f, 0.05f,0, 0.7f,0.6f, +1.0f, -0.70f);
            break; }
        case 3:{ // major galaxy + infalling satellite (minor merger)
            int sat=n/5, big=n-sat;
            addDisk(0,big,  0,0,0, 0,0,0, 1.0f,1.0f, +1.0f, 0.30f);
            addDisk(big,sat, 1.4f,0.7f,0.1f, -0.25f,-0.15f,0, 0.2f,0.35f, -1.0f, 0.90f);
            gasDiskFrac(0.8f, 0,0,0, 0,0,0, 1.0f,1.0f, +1.0f, 0.30f);
            gasDiskFrac(0.2f, 1.4f,0.7f,0.1f, -0.25f,-0.15f,0, 0.2f,0.35f, -1.0f, 0.90f);
            break; }
        case 4:  // cold collapse (sphere falls in, fragments)
            addSphere(0,n, 0,0,0, 1.0f, 1.0f, 0.0f, 0.05f);
            gasSphereFrac(1.0f, 0,0,0, 1.0f, 0.0f, 0.05f);
            break;
        case 5:  // expanding cloud (outward velocity)
            addSphere(0,n, 0,0,0, 0.4f, 1.0f, 0.9f, 0.10f);
            gasSphereFrac(1.0f, 0,0,0, 0.4f, 0.9f, 0.10f);
            break;
        case 6:{ // head-on collision, perpendicular disks
            int h=n/2;
            addDisk(0,h,   -1.0f,0,0,  0.30f,0,0, 0.7f,0.6f, +1.0f, 0.0f);
            addDisk(h,n-h,  1.0f,0,0, -0.30f,0,0, 0.7f,0.6f, -1.0f, 1.5708f);
            gasDiskFrac(0.5f, -1.0f,0,0,  0.30f,0,0, 0.7f,0.6f, +1.0f, 0.0f);
            gasDiskFrac(0.5f,  1.0f,0,0, -0.30f,0,0, 0.7f,0.6f, -1.0f, 1.5708f);
            break; }
        case 7:{ // BIG BANG — cosmological structure / galaxy formation
            // Near-uniform sphere with Zel'dovich density perturbations and a
            // Hubble expansion velocity field. Gravity grows the perturbations:
            // matter drains out of voids into sheets -> filaments -> collapsed
            // haloes (proto-galaxies) that merge hierarchically. H0 is set just
            // below the binding value so the box expands, turns around, and the
            // cosmic web condenses out of it.
            const float R0   = 320.0f;                     // half-extent of the cube (huge box, wide spacing)
            const float Mtot = 1.0f;                       // total (stars carry it)
            // Hc = the H0 at which this box's actual matter density equals the
            // critical density (Omega_m=1). Real LCDM is sub-critical in matter
            // (Omega_m~0.3) and dominated today by dark energy (Omega_L~0.7):
            // H0 = Hc/sqrt(Omega_m) reproduces that for the given box mass, and
            // the Lambda term below adds the actual accelerating-expansion
            // physics (Friedmann eq.: a'' = -GM/r^2 + Omega_L*H0^2*r) instead of
            // just tuning H0 by hand to "looks about right".
            const float Hc   = sqrtf(2.0f*G*Mtot/(R0*R0*R0));
            const float H0   = Hc/sqrtf(fmaxf(p.omegaM,1e-4f));
            lambdaTerm_ = p.omegaL*H0*H0;
            const float PI   = 3.14159265f;

            // This box's dynamical times dwarf the galaxy presets': the Hubble
            // time is 1/H0 ~ 2200 sim units and the mean-density free-fall time
            // ~ 9000, so the default dt=8e-4 would need millions of steps before
            // the web condenses -- it looks like "nothing forms". Tie dt to the
            // expansion rate instead (a few thousand steps per Hubble time),
            // unless the user pinned it with --dt.
            if (!p.dtGiven) p_.dt = 1.0f/(3600.0f*H0);
            // Likewise soften to a fraction of the mean interparticle spacing
            // (~6 units at 1M bodies in a 640-unit box). The galaxy-scale
            // eps=0.004 is ~1000x too hard here: collapsed knots would two-body
            // scatter and evaporate instead of surviving as haloes/proto-galaxies.
            {
                float spacing = 2.0f*R0/cbrtf((float)(nS>0?nS:1));
                p_.eps = 0.08f*spacing;
            }
            std::printf("Big Bang: H0=%.3e  dt=%.3f (%s)  eps=%.3f  (t_H ~ %.0f sim units)\n",
                H0, p_.dt, p.dtGiven?"--dt":"auto", p_.eps, 1.0f/H0);

            // --- Gaussian random displacement field with a ΛCDM-like P(k) ---
            // Real cosmic-web structure spans many scales: huge voids bounded by
            // sheets, sheets draining into filaments, filaments knotting into
            // haloes. A handful of long-wavelength modes only makes a few blobs,
            // so we sum many modes log-sampled across a decade of wavenumber with
            // a red-tilted, small-scale-damped spectrum (more power on large
            // scales, gently cut at small scales like the CDM transfer function).
            // NM modes log-spaced across a WIDE range of scales: too narrow a
            // range (or too steep a tilt) lets the single box-fundamental mode
            // dominate, which renders as one giant X-shaped filament crossing
            // the whole box instead of a proper tiled web of many voids/cells
            // (like a slice through a real cosmological box). Starting above
            // the fundamental and using a shallower tilt (k^-0.7 instead of
            // k^-1) spreads the power across several octaves so dozens of
            // independent voids/filaments/knots form across the volume.
            const int   NM     = 96;
            const float kfund  = PI / R0;                  // box fundamental (wavelength 2*R0)
            const float kmin   = 1.4f * kfund;              // mildly suppress the single global mode
            const float kmax   = 14.0f * kfund;             // fine filament/knot scale
            const float kcut   = 9.0f * kfund;              // small-scale damping knee
            const float tilt   = 0.9f;                      // spectral slope amp ~ k^-tilt
            float kx[NM],ky[NM],kz[NM],amp[NM],ph[NM],kl[NM];
            float sumA2 = 0.0f;
            for(int m=0;m<NM;++m){
                // isotropic random direction
                float dx=2*uni(rng)-1, dy=2*uni(rng)-1, dz=2*uni(rng)-1;
                float dl=sqrtf(dx*dx+dy*dy+dz*dz)+1e-4f;
                // log-uniform |k| -> even coverage of every scale
                float kmag=kmin*powf(kmax/kmin, uni(rng));
                kx[m]=dx/dl*kmag; ky[m]=dy/dl*kmag; kz[m]=dz/dl*kmag; kl[m]=kmag;
                // red spectrum with a Gaussian small-scale cutoff + random
                // mode-to-mode amplitude scatter (Rayleigh-ish) for irregularity
                float damp = expf(-(kmag*kmag)/(kcut*kcut));
                float a = powf(kmin/kmag, tilt) * damp * (0.5f + uni(rng));
                amp[m]=a; sumA2 += a*a;
                ph[m]=uni(rng)*6.2831853f;
            }
            // Normalize so the RMS Zel'dovich displacement is a fixed fraction of
            // R0 no matter how many modes we use (keeps the field strong but below
            // runaway shell-crossing). rms|d| = sqrt(sum amp^2 / 2) before scaling.
            const float targetRMS = 0.24f * R0;            // structure strength
            float gscale = (sumA2>1e-12f) ? targetRMS/sqrtf(0.5f*sumA2) : 0.0f;
            for(int m=0;m<NM;++m) amp[m]*=gscale;
            // Zel'dovich growing mode: v_pec = f*H0*d, with the linear growth
            // rate f = dlnD/dlna ~ Omega_m^0.55. The coupling has units of
            // 1/time (it multiplies a displacement), so it MUST scale with H0
            // (~4.5e-4 here) -- a hard-coded O(1) value gives peculiar speeds
            // hundreds of times the escape speed and free-streams the box into
            // a structureless haze instead of letting the web grow.
            const float vpec = powf(fmaxf(p.omegaM,1e-4f),0.55f)*H0;

            auto seedBB=[&](int s,int cnt,float mPart){
                for(int k=0;k<cnt;++k){ int i=s+k;
                    // uniform fill of a big cube [-R0,R0]^3 (no rejection)
                    float qx=2*uni(rng)-1, qy=2*uni(rng)-1, qz=2*uni(rng)-1;
                    qx*=R0; qy*=R0; qz*=R0;
                    // Zel'dovich displacement = gradient of the potential field:
                    // d = sum_m amp_m cos(k.q+phi) k_hat   (cos so d and its
                    // velocity stay in phase for a clean growing mode)
                    float px=0,py=0,pz=0;
                    for(int m=0;m<NM;++m){
                        float a=amp[m]*cosf(kx[m]*qx+ky[m]*qy+kz[m]*qz+ph[m]);
                        px+=a*kx[m]/kl[m]; py+=a*ky[m]/kl[m]; pz+=a*kz[m]/kl[m];
                    }
                    float x=qx+px, y=qy+py, z=qz+pz;
                    hx[i]=x; hy[i]=y; hz[i]=z;
                    hvx[i]=H0*x + vpec*px;                          // Hubble + growing mode
                    hvy[i]=H0*y + vpec*py;
                    hvz[i]=H0*z + vpec*pz;
                    hm[i]=mPart;
                }
            };
            seedBB(0, nS, Mtot/(float)(nS>0?nS:1));                 // stars carry the mass
            seedBB(gB, nG, mTracer);                                // gas + dust trace it
            seedBB(dB, nD, mTracer);
            go=nG; doff=nD;                                         // ranges fully seeded
            break; }
    }
    // Any rounding remainder in the gas/dust ranges: clone the last tracer so
    // there are no uninitialised bodies in the arrays.
    for(int i=gB+go; i<gB+nG; ++i){ hx[i]=hx[gB]; hy[i]=hy[gB]; hz[i]=hz[gB];
        hvx[i]=hvx[gB]; hvy[i]=hvy[gB]; hvz[i]=hvz[gB]; hm[i]=mTracer; }
    for(int i=dB+doff; i<dB+nD; ++i){ hx[i]=hx[dB]; hy[i]=hy[dB]; hz[i]=hz[dB];
        hvx[i]=hvx[dB]; hvy[i]=hvy[dB]; hvz[i]=hvz[dB]; hm[i]=mTracer; }
    CUDA_CHECK(cudaMemcpy(posx_,hx.data(),fb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(posy_,hy.data(),fb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(posz_,hz.data(),fb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(velx_,hvx.data(),fb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vely_,hvy.data(),fb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(velz_,hvz.data(),fb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mass_,hm.data(),fb,cudaMemcpyHostToDevice));
    worldRadius_ = 1.2f;

    // Upload the static halo table + cosmological Lambda term to constant memory.
    int haloCount = (int)haloX.size();
    CUDA_CHECK(cudaMemcpyToSymbol(d_haloCount, &haloCount, sizeof(int)));
    if (haloCount > 0) {
        CUDA_CHECK(cudaMemcpyToSymbol(d_haloX, haloX.data(), sizeof(float)*haloCount));
        CUDA_CHECK(cudaMemcpyToSymbol(d_haloY, haloY.data(), sizeof(float)*haloCount));
        CUDA_CHECK(cudaMemcpyToSymbol(d_haloZ, haloZ.data(), sizeof(float)*haloCount));
        CUDA_CHECK(cudaMemcpyToSymbol(d_haloMs, haloMs.data(), sizeof(float)*haloCount));
        CUDA_CHECK(cudaMemcpyToSymbol(d_haloRs, haloRs.data(), sizeof(float)*haloCount));
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_lambdaTerm, &lambdaTerm_, sizeof(float)));

    // Prime accx_/accy_/accz_ with a(t0) so step()'s leapfrog kick-drift-kick
    // has a valid initial half-kick on the very first call.
    buildTree(); summarize(); computeForces(); applyExternalForces();
}

Simulation::~Simulation() {
    cudaFree(posx_);cudaFree(posy_);cudaFree(posz_);
    cudaFree(velx_);cudaFree(vely_);cudaFree(velz_);
    cudaFree(accx_);cudaFree(accy_);cudaFree(accz_);
    cudaFree(mass_);
    cudaFree(ncom_);cudaFree(nsize2_);
    cudaFree(nminx_);cudaFree(nminy_);cudaFree(nminz_);
    cudaFree(nmaxx_);cudaFree(nmaxy_);cudaFree(nmaxz_);
    if (nquadA_) { cudaFree(nquadA_); cudaFree(nquadB_); }
    cudaFree(left_);cudaFree(right_);cudaFree(parent_);cudaFree(visited_);
    cudaFree(code_);cudaFree(idx_);
}

void Simulation::centerOfMass(float& x, float& y, float& z) {
    if(!treeBuilt_){ x=y=z=0.0f; return; }
    // Root internal node (id 0) carries the total mass-weighted COM.
    float4 com;
    cudaMemcpy(&com, ncom_, sizeof(float4), cudaMemcpyDeviceToHost);
    x = com.x; y = com.y; z = com.z;
}

void Simulation::buildTree() {
    treeBuilt_ = true;
    int gn = grid(nbodies_);
    initBoundsKernel<<<1,1>>>();
    boundsKernel<<<gn,BLOCK>>>(posx_,posy_,posz_,nbodies_);
    setCubeKernel<<<1,1>>>();
    mortonKernel<<<gn,BLOCK>>>(posx_,posy_,posz_,code_,idx_,nbodies_);

    thrust::device_ptr<unsigned> kc(code_);
    thrust::device_ptr<int>      ki(idx_);
    thrust::sort_by_key(kc, kc + nbodies_, ki);

    buildTreeKernel<<<gn,BLOCK>>>(code_,nbodies_,left_,right_,parent_);
    initLeavesKernel<<<gn,BLOCK>>>(posx_,posy_,posz_,mass_,idx_,nbodies_,
        ncom_,nsize2_,nminx_,nminy_,nminz_,nmaxx_,nmaxy_,nmaxz_,nquadA_,nquadB_,visited_);
}

void Simulation::summarize() {
    summarizeKernel<<<grid(nbodies_),BLOCK>>>(parent_,left_,right_,nbodies_,
        ncom_,nsize2_,nminx_,nminy_,nminz_,nmaxx_,nmaxy_,nmaxz_,nquadA_,nquadB_,visited_);
}

void Simulation::computeForces() {
    forcesKernel<<<grid(nbodies_),BLOCK>>>(posx_,posy_,posz_,idx_,left_,right_,
        ncom_,nsize2_,nquadA_,nquadB_,accx_,accy_,accz_,
        nbodies_,p_.theta,p_.eps*p_.eps,p_.G,p_.useQuadrupole);
}

void Simulation::applyExternalForces() {
    externalAccelKernel<<<grid(nbodies_),BLOCK>>>(posx_,posy_,posz_,accx_,accy_,accz_,nbodies_,p_.G);
}

void Simulation::computeAccelOnly() {
    // Pure N-body tree force (no halo/Lambda) -- used for diagnostics/--verify,
    // where we compare against an exact O(N^2) N-body reference.
    buildTree(); summarize();
    computeForces();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Simulation::step() {
    // Leapfrog kick-drift-kick: accx_/accy_/accz_ already hold a(t) from the
    // end of the previous step (or the constructor's initial force eval).
    float dtHalf = 0.5f*p_.dt;
    float dampHalf = sqrtf(p_.damping);
    kickKernel<<<grid(nbodies_),BLOCK>>>(velx_,vely_,velz_,accx_,accy_,accz_,nbodies_,dtHalf,dampHalf);
    driftKernel<<<grid(nbodies_),BLOCK>>>(posx_,posy_,posz_,velx_,vely_,velz_,nbodies_,p_.dt);
    buildTree(); summarize();
    computeForces();
    applyExternalForces();          // a(t+dt) now includes halo(s) + Lambda term
    kickKernel<<<grid(nbodies_),BLOCK>>>(velx_,vely_,velz_,accx_,accy_,accz_,nbodies_,dtHalf,dampHalf);
}

void Simulation::copyToRenderBuffer(float* d_dst) {
    renderCopyKernel<<<grid(nbodies_),BLOCK>>>(posx_,posy_,posz_,velx_,vely_,velz_,d_dst,nbodies_);
}

void Simulation::computeDirectReference(float* d_ax, float* d_ay, float* d_az) {
    directKernel<<<grid(nbodies_),BLOCK>>>(posx_,posy_,posz_,mass_,d_ax,d_ay,d_az,
        nbodies_,p_.eps*p_.eps,p_.G);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Simulation::accelError(const float* d_refx, const float* d_refy, const float* d_refz,
                            double& meanRelErr, double& maxRelErr) {
    std::vector<float> ax(nbodies_),ay(nbodies_),az(nbodies_),rx(nbodies_),ry(nbodies_),rz(nbodies_);
    CUDA_CHECK(cudaMemcpy(ax.data(),accx_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay.data(),accy_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az.data(),accz_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rx.data(),d_refx,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ry.data(),d_refy,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rz.data(),d_refz,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    double sum=0; maxRelErr=0;
    for (int i=0;i<nbodies_;++i){
        double ex=ax[i]-rx[i],ey=ay[i]-ry[i],ez=az[i]-rz[i];
        double err=std::sqrt(ex*ex+ey*ey+ez*ez);
        double mag=std::sqrt((double)rx[i]*rx[i]+(double)ry[i]*ry[i]+(double)rz[i]*rz[i])+1e-12;
        double rel=err/mag; sum+=rel; if(rel>maxRelErr)maxRelErr=rel;
    }
    meanRelErr=sum/nbodies_;
}

void Simulation::dumpPositions(const char* path) {
    std::vector<float> x(nbodies_),y(nbodies_),z(nbodies_);
    CUDA_CHECK(cudaMemcpy(x.data(),posx_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(),posy_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z.data(),posz_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    FILE* f = std::fopen(path,"w");
    if(!f){ std::fprintf(stderr,"dump: cannot open %s\n",path); return; }
    for(int i=0;i<nbodies_;++i) std::fprintf(f,"%g %g %g\n",x[i],y[i],z[i]);
    std::fclose(f);
    std::printf("dumped %d positions -> %s\n", nbodies_, path);
}

double Simulation::totalEnergy() {
    std::vector<float> x(nbodies_),y(nbodies_),z(nbodies_),vx(nbodies_),vy(nbodies_),vz(nbodies_),m(nbodies_);
    CUDA_CHECK(cudaMemcpy(x.data(),posx_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(),posy_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z.data(),posz_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(),velx_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy.data(),vely_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz.data(),velz_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m.data(),mass_,sizeof(float)*nbodies_,cudaMemcpyDeviceToHost));
    double ke=0,pe=0,eps2=(double)p_.eps*p_.eps;
    for(int i=0;i<nbodies_;++i) ke+=0.5*m[i]*((double)vx[i]*vx[i]+(double)vy[i]*vy[i]+(double)vz[i]*vz[i]);
    for(int i=0;i<nbodies_;++i) for(int j=i+1;j<nbodies_;++j){
        double dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i];
        pe-=p_.G*m[i]*m[j]/std::sqrt(dx*dx+dy*dy+dz*dz+eps2);
    }
    return ke+pe;
}
