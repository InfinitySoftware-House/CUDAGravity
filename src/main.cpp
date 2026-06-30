// 3D CUDA Barnes-Hut N-body — HDR OpenGL renderer + headless test modes.
//
//   gravity                 interactive 3D viewer (default 100k bodies)
//   gravity --n=200000      choose body count
//   gravity --bench         headless: time N steps, print ms/step and FPS
//   gravity --verify        compare Barnes-Hut accel against brute force O(N^2)
//
// Viewer controls:
//   left-drag orbit, scroll zoom, R reset, Space pause, A auto-spin,
//   V toggle velocity colouring (blue=slow -> red=fast), C record video, Esc quit.
//
// Rendering: stars are drawn as additive sprites into an HDR (RGBA16F) buffer,
// then bloomed and tone-mapped (ACES) for realistic brightness. Each star has
// an intrinsic black-body colour and luminosity.
//
// Recording pipes tone-mapped frames to ffmpeg at a fixed 60 fps, so the output
// is always smooth 60 fps regardless of how fast the simulation renders.

#include "sim.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>

// ---------------------------------------------------------------------------
//  Headless modes (no OpenGL needed)
// ---------------------------------------------------------------------------
static int runBench(SimParams p, int steps) {
    std::printf("Benchmark: %d bodies, %d steps (theta=%.2f)\n", p.n, steps, p.theta);
    Simulation sim(p);
    cudaDeviceSynchronize();
    for (int i = 0; i < 5; ++i) sim.step();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < steps; ++i) sim.step();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double msPer = ms / steps;
    std::printf("Total %.1f ms  ->  %.3f ms/step  =  %.1f FPS\n", ms, msPer, 1000.0/msPer);
    std::printf("Throughput: %.1f million body-steps/sec\n", (double)p.n*steps/(ms/1000.0)/1e6);
    return 0;
}
static int runVerify(SimParams p) {
    if (p.n > 16384) { p.n = 8192; std::printf("verify: clamping N to %d\n", p.n); }
    std::printf("Verify: %d bodies, theta=%.2f vs brute force\n", p.n, p.theta);
    Simulation sim(p);
    float *rx,*ry,*rz; cudaMalloc(&rx,sizeof(float)*p.n); cudaMalloc(&ry,sizeof(float)*p.n); cudaMalloc(&rz,sizeof(float)*p.n);
    sim.computeAccelOnly();
    sim.computeDirectReference(rx,ry,rz);
    double meanErr,maxErr; sim.accelError(rx,ry,rz,meanErr,maxErr);
    std::printf("Relative accel error vs O(N^2):  mean=%.3e  max=%.3e\n", meanErr, maxErr);
    std::printf("%s\n", meanErr < 0.02 ? "PASS (mean < 2%)" : "WARN (mean >= 2%)");
    cudaFree(rx); cudaFree(ry); cudaFree(rz);
    return meanErr < 0.05 ? 0 : 1;
}

// ---------------------------------------------------------------------------
//  Interactive OpenGL viewer
// ---------------------------------------------------------------------------
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

// --- minimal 4x4 matrix math (column-major, GL style) ---
struct Mat4 { float m[16]; };
static Mat4 identity(){ Mat4 r{}; for(int i=0;i<4;++i) r.m[i*4+i]=1.0f; return r; }
static Mat4 mul(const Mat4&a,const Mat4&b){ Mat4 r{};
    for(int c=0;c<4;++c) for(int row=0;row<4;++row){ float s=0; for(int k=0;k<4;++k) s+=a.m[k*4+row]*b.m[c*4+k]; r.m[c*4+row]=s; } return r; }
static Mat4 perspective(float fovy,float asp,float zn,float zf){ Mat4 r{}; float f=1.0f/std::tan(fovy*0.5f);
    r.m[0]=f/asp; r.m[5]=f; r.m[10]=(zf+zn)/(zn-zf); r.m[11]=-1.0f; r.m[14]=(2.0f*zf*zn)/(zn-zf); return r; }
static Mat4 lookAt(float ex,float ey,float ez,float cx,float cy,float cz,float ux,float uy,float uz){
    float fx=cx-ex,fy=cy-ey,fz=cz-ez,fl=std::sqrt(fx*fx+fy*fy+fz*fz); fx/=fl;fy/=fl;fz/=fl;
    float sx=fy*uz-fz*uy,sy=fz*ux-fx*uz,sz=fx*uy-fy*ux,sl=std::sqrt(sx*sx+sy*sy+sz*sz); sx/=sl;sy/=sl;sz/=sl;
    float ux2=sy*fz-sz*fy,uy2=sz*fx-sx*fz,uz2=sx*fy-sy*fx; Mat4 r=identity();
    r.m[0]=sx;r.m[4]=sy;r.m[8]=sz; r.m[1]=ux2;r.m[5]=uy2;r.m[9]=uz2; r.m[2]=-fx;r.m[6]=-fy;r.m[10]=-fz;
    r.m[12]=-(sx*ex+sy*ey+sz*ez); r.m[13]=-(ux2*ex+uy2*ey+uz2*ez); r.m[14]=(fx*ex+fy*ey+fz*ez); return r; }

// --- free-fly camera / ui state ---
// Orientation is kept as an orthonormal basis (forward/right/up) and rotated
// incrementally about the camera's own axes -> full 360 freedom, no gimbal lock.
static float g_camPos[3]={1.6f,1.3f,2.2f};
static float g_f[3]={0,0,-1}, g_r[3]={1,0,0}, g_u[3]={0,1,0};
static bool  g_dragging=false, g_paused=false, g_autospin=false;
static int   g_mode=0;                 // 0 = realistic, 1 = velocity
static bool  g_toggleRecord=false;     // edge flag set by key
static int   g_preset=1;               // active IC template (1..6)
static bool  g_restart=false;          // edge flag: rebuild sim with g_preset
static bool  g_clouds=true;            // render ambient nebulae + dust
static bool  g_orbit=false;            // auto-orbit the centre of mass
static float g_orbAz=0.0f, g_orbEl=0.35f, g_orbR=4.0f;   // orbit angle/elev/radius
static float g_orbSpeed=0.0035f;       // radians per frame
static double g_lastX=0, g_lastY=0;

static void vnorm(float*v){ float l=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if(l>1e-8f){v[0]/=l;v[1]/=l;v[2]/=l;} }
static void vcross(const float*a,const float*b,float*o){ o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0]; }
// rotate v about unit axis k by angle ang (Rodrigues), in place
static void vrot(float*v,const float*k,float ang){
    float c=std::cos(ang), s=std::sin(ang);
    float kv=k[0]*v[0]+k[1]*v[1]+k[2]*v[2], cx[3]; vcross(k,v,cx);
    for(int i=0;i<3;++i) v[i]=v[i]*c + cx[i]*s + k[i]*kv*(1.0f-c);
}
static void camReset(){
    // Big Bang fills a much larger volume (R0~3.2) than the galaxy presets, so
    // start the camera further back to frame the whole forming cosmic web.
    float d = (g_preset==7) ? 8.0f : 1.0f;
    g_camPos[0]=1.6f*d; g_camPos[1]=1.3f*d; g_camPos[2]=2.2f*d;
    float f[3]={-g_camPos[0],-g_camPos[1],-g_camPos[2]}; vnorm(f);
    float wup[3]={0,1,0}, r[3]; vcross(f,wup,r); vnorm(r);
    float u[3]; vcross(r,f,u);
    for(int i=0;i<3;++i){ g_f[i]=f[i]; g_r[i]=r[i]; g_u[i]=u[i]; }
}

static void onMouseButton(GLFWwindow*w,int b,int act,int){ if(b==GLFW_MOUSE_BUTTON_LEFT){ g_dragging=(act==GLFW_PRESS); glfwGetCursorPos(w,&g_lastX,&g_lastY);} }
static void onCursor(GLFWwindow*,double x,double y){
    if(!g_dragging){ g_lastX=x; g_lastY=y; return; }
    float dx=(float)(x-g_lastX), dy=(float)(y-g_lastY);
    g_lastX=x; g_lastY=y;
    const float sens=0.005f;                       // drag-to-look (no pitch clamp)
    vrot(g_f,g_u,-dx*sens); vrot(g_r,g_u,-dx*sens);
    vrot(g_f,g_r,-dy*sens); vrot(g_u,g_r,-dy*sens);
    vnorm(g_f); vnorm(g_r); vnorm(g_u);
}
static void onScroll(GLFWwindow*,double,double dy){      // dolly along view direction
    for(int i=0;i<3;++i) g_camPos[i]+=g_f[i]*(float)dy*0.15f;
}
static void onKey(GLFWwindow*w,int key,int,int act,int){ if(act!=GLFW_PRESS)return;
    if(key==GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w,1);
    if(key==GLFW_KEY_SPACE)  g_paused=!g_paused;
    if(key==GLFW_KEY_X)      g_autospin=!g_autospin;
    if(key==GLFW_KEY_V)      g_mode^=1;
    if(key==GLFW_KEY_C)      g_toggleRecord=true;
    if(key==GLFW_KEY_N)      g_clouds=!g_clouds;
    if(key==GLFW_KEY_O)      g_orbit=!g_orbit;
    if(key>=GLFW_KEY_1 && key<=GLFW_KEY_7){ g_preset=key-GLFW_KEY_0; g_restart=true; }
    if(key==GLFW_KEY_R)      camReset(); }

static GLuint compile(GLenum type,const char*src){
    GLuint s=glCreateShader(type); glShaderSource(s,1,&src,nullptr); glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ char log[2048]; glGetShaderInfoLog(s,2048,nullptr,log); std::fprintf(stderr,"shader error:\n%s\n",log); std::exit(1);} return s; }
static GLuint program(const char*vs,const char*fs){
    GLuint v=compile(GL_VERTEX_SHADER,vs), f=compile(GL_FRAGMENT_SHADER,fs);
    GLuint p=glCreateProgram(); glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,nullptr,log); std::fprintf(stderr,"link error:\n%s\n",log); std::exit(1);}
    glDeleteShader(v); glDeleteShader(f); return p; }

// ---- shaders ----
static const char* kStarVert = R"(#version 330 core
layout(location=0) in vec4 aData;   // xyz position, w = speed
layout(location=1) in vec2 aStar;   // x = temperature[0..1], y = luminosity
uniform mat4  uMVP;
uniform float uPointScale;
uniform int   uMode;                // 0 realistic, 1 velocity
uniform float uSpeedScale;
out vec3 vColor;
out float vBright;

vec3 blackbody(float t){            // t: 0 cool/red .. 1 hot/blue
    vec3 red  = vec3(1.00, 0.50, 0.28);
    vec3 white= vec3(1.00, 0.96, 0.90);
    vec3 blue = vec3(0.66, 0.78, 1.00);
    return t < 0.5 ? mix(red, white, t*2.0) : mix(white, blue, (t-0.5)*2.0);
}
vec3 velmap(float s){               // blue -> cyan -> green -> yellow -> red
    vec3 c0=vec3(0.10,0.20,1.00), c1=vec3(0.00,0.90,1.00), c2=vec3(0.10,1.00,0.20),
         c3=vec3(1.00,0.95,0.00), c4=vec3(1.00,0.10,0.05);
    if(s<0.25) return mix(c0,c1,s/0.25);
    if(s<0.50) return mix(c1,c2,(s-0.25)/0.25);
    if(s<0.75) return mix(c2,c3,(s-0.50)/0.25);
    return mix(c3,c4,(s-0.75)/0.25);
}
void main(){
    vec4 clip = uMVP * vec4(aData.xyz, 1.0);
    gl_Position = clip;
    float lum = aStar.y;
    gl_PointSize = clamp(uPointScale * sqrt(lum) / clip.w, 1.0, 60.0);
    if(uMode == 0){ vColor = blackbody(aStar.x); vBright = lum; }
    else { float s = clamp(aData.w * uSpeedScale, 0.0, 1.0); vColor = velmap(s); vBright = 1.1; }
}
)";

static const char* kStarFrag = R"(#version 330 core
in vec3 vColor; in float vBright;
out vec4 FragColor;
void main(){
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d,d);
    if(r2 > 0.25) discard;
    float core = exp(-r2 * 22.0);          // tight bright core
    float halo = exp(-r2 * 6.0) * 0.12;     // faint surrounding glow
    float a = core + halo;
    FragColor = vec4(vColor * vBright * a * 1.5, a);   // HDR, additive
}
)";

// ---- nebula / dust sprites ----
// The gas/dust are real simulated tracer bodies; here they are drawn as small
// soft sprites. Emission colour comes from a coherent 3D value-noise field in
// world space, so neighbouring parcels share a hue -> believable nebula regions.
// Attribs: vec4 pos+speed (shared CUDA buffer), vec2 (size, brightness).
static const char* kCloudVert = R"(#version 330 core
layout(location=0) in vec4  aData;  // xyz position, w = speed (shared with stars)
layout(location=1) in vec2  aAttr;  // x = world-space radius, y = brightness/opacity
uniform mat4  uMVP;
uniform float uScale;
out vec3  vWorld;
out float vW;
void main(){
    vec4 clip = uMVP * vec4(aData.xyz, 1.0);
    gl_Position  = clip;
    gl_PointSize = clamp(uScale * aAttr.x / max(clip.w, 1e-3), 1.0, 220.0);
    vWorld = aData.xyz; vW = aAttr.y;
}
)";
// shared GLSL value-noise (iq-style) used to colour/modulate the gas
static const char* kNoiseGLSL = R"(
float hash(vec3 p){ p = fract(p*0.3183099 + vec3(0.1,0.2,0.3));
    p *= 17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
float vnoise(vec3 x){
    vec3 i = floor(x), f = fract(x); f = f*f*(3.0-2.0*f);
    return mix(mix(mix(hash(i+vec3(0,0,0)),hash(i+vec3(1,0,0)),f.x),
                   mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),
               mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),
                   mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}
vec3 emission(vec3 p){
    vec3 pal[6] = vec3[6](
        vec3(1.00,0.18,0.30), vec3(0.95,0.30,0.78), vec3(0.45,0.35,1.00),
        vec3(0.20,0.70,1.00), vec3(0.15,0.90,0.65), vec3(1.00,0.62,0.22));
    float t = vnoise(p*1.6 + vec3(11.0,3.0,7.0)) * 5.0;
    int i = int(t); float f = fract(t); int j = (i+1) % 6;
    vec3 c = mix(pal[i], pal[j], f);
    return c * (0.75 + 0.5*vnoise(p*3.7 + vec3(-5.0,9.0,-2.0)));
}
)";
// Emissive gas: additive, soft gaussian falloff. Colour from the world field.
static const char* kNebulaFrag = R"(#version 330 core
in vec3 vWorld; in float vW; out vec4 o;
%NOISE%
void main(){
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d,d);
    if(r2 > 0.25) discard;
    float a = exp(-r2 * 7.0) * (1.0 - smoothstep(0.18, 0.25, r2));
    o = vec4(emission(vWorld) * vW * a, a);   // HDR additive (blend ONE,ONE)
}
)";
// Dust: dark, slightly warm tint, alpha-over so it carves dark lanes.
static const char* kDustFrag = R"(#version 330 core
in vec3 vWorld; in float vW; out vec4 o;
%NOISE%
void main(){
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d,d);
    if(r2 > 0.25) discard;
    float a = exp(-r2 * 5.0) * (1.0 - smoothstep(0.16, 0.25, r2));
    float w = vnoise(vWorld*1.7);
    vec3 tint = vec3(0.05*(1.0+0.9*w), 0.038, 0.030);
    o = vec4(tint, a * vW);                    // blend SRC_ALPHA, ONE_MINUS_SRC_ALPHA
}
)";

static const char* kFsTriVert = R"(#version 330 core
out vec2 uv;
void main(){
    vec2 p = vec2((gl_VertexID==2)?3.0:-1.0, (gl_VertexID==1)?3.0:-1.0);
    uv = p*0.5+0.5;
    gl_Position = vec4(p,0.0,1.0);
}
)";
static const char* kBrightFrag = R"(#version 330 core
in vec2 uv; out vec4 o; uniform sampler2D uScene; uniform float uThresh;
void main(){ vec3 c=texture(uScene,uv).rgb; float l=dot(c,vec3(0.2126,0.7152,0.0722));
    o = vec4(c * max(l-uThresh,0.0)/max(l,1e-4), 1.0); }
)";
static const char* kBlurFrag = R"(#version 330 core
in vec2 uv; out vec4 o; uniform sampler2D uTex; uniform vec2 uDir;
void main(){
    float w[5]=float[](0.227027,0.194595,0.121622,0.054054,0.016216);
    vec3 c = texture(uTex,uv).rgb * w[0];
    for(int i=1;i<5;++i){ c += texture(uTex, uv + uDir*float(i)).rgb * w[i];
                          c += texture(uTex, uv - uDir*float(i)).rgb * w[i]; }
    o = vec4(c,1.0);
}
)";
static const char* kCompositeFrag = R"(#version 330 core
in vec2 uv; out vec4 o;
uniform sampler2D uScene; uniform sampler2D uBloom;
uniform float uExposure; uniform float uBloomAmt;
vec3 aces(vec3 x){ const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0); }
void main(){
    vec3 hdr = texture(uScene,uv).rgb + texture(uBloom,uv).rgb * uBloomAmt;
    vec3 col = aces(hdr * uExposure);
    col = pow(col, vec3(1.0/2.2));
    o = vec4(col,1.0);
}
)";

// ---- simple HDR framebuffer helper ----
struct FBO { GLuint fbo=0, tex=0; int w=0,h=0; };
static void makeFBO(FBO& f,int w,int h){
    if(f.fbo) glDeleteFramebuffers(1,&f.fbo);
    if(f.tex) glDeleteTextures(1,&f.tex);
    f.w=w; f.h=h;
    glGenTextures(1,&f.tex); glBindTexture(GL_TEXTURE_2D,f.tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1,&f.fbo); glBindFramebuffer(GL_FRAMEBUFFER,f.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,f.tex,0);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}
// 8-bit LDR target used as the video-capture render destination.
static void makeFBO8(FBO& f,int w,int h){
    if(f.fbo) glDeleteFramebuffers(1,&f.fbo);
    if(f.tex) glDeleteTextures(1,&f.tex);
    f.w=w; f.h=h;
    glGenTextures(1,&f.tex); glBindTexture(GL_TEXTURE_2D,f.tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1,&f.fbo); glBindFramebuffer(GL_FRAMEBUFFER,f.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,f.tex,0);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}

static bool ffmpegAvailable(){ return std::system("ffmpeg -version >nul 2>&1") == 0; }

// Compile a program after splicing the shared noise GLSL into the "%NOISE%"
// marker of the fragment source.
static GLuint programNoise(const char* vs, const char* fsTemplate){
    std::string fs = fsTemplate;
    const std::string mark = "%NOISE%";
    size_t at = fs.find(mark);
    if(at != std::string::npos) fs.replace(at, mark.size(), kNoiseGLSL);
    return program(vs, fs.c_str());
}

// Per-body cloud attributes (size, brightness) for the gas/dust tracer ranges.
// Stars [0,nStars) are left zero (never drawn by the cloud passes). The spatial
// structure of the nebulae comes from the *simulated* particle density; this
// only sets each sprite's footprint and intensity, kept small and dim so the
// gas reads as a soft haze rather than opaque blobs.
static void buildCloudAttrs(int nStars,int nGas,int nDust, std::vector<float>& attr){
    int total = nStars + nGas + nDust;
    attr.assign((size_t)total*2, 0.0f);
    std::mt19937 rng(7u); std::uniform_real_distribution<float> uni(0,1);
    for(int i=nStars; i<nStars+nGas; ++i){
        attr[2*i+0] = 0.014f + 0.026f*uni(rng);     // world-space radius (small)
        attr[2*i+1] = 0.010f + 0.020f*uni(rng);     // additive brightness (dim)
    }
    for(int i=nStars+nGas; i<total; ++i){
        attr[2*i+0] = 0.010f + 0.020f*uni(rng);     // radius
        attr[2*i+1] = 0.035f + 0.085f*uni(rng);     // opacity
    }
}

static int runViewer(SimParams p, int cliN, int cliGas, int cliDust,
                     bool autoRecord, int autoRecFrames,
                     bool headless, int recFps, int subSteps,
                     int capW, int capH, int recCrf, bool orbitOn) {
    if(!glfwInit()){ std::fprintf(stderr,"glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    // Headless: hidden window -> offscreen context, no compositor, runs at full
    // GPU speed (no vsync). Used for fast, non-realtime video rendering.
    if(headless) glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
    GLFWwindow* win=glfwCreateWindow(1280,800,"CUDA Barnes-Hut 3D N-body",nullptr,nullptr);
    if(!win){ std::fprintf(stderr,"window creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win); glfwSwapInterval(headless?0:1);   // no vsync when headless
    if(!gladLoadGL((GLADloadfunc)glfwGetProcAddress)){ std::fprintf(stderr,"GLAD load failed\n"); return 1; }
    glfwSetMouseButtonCallback(win,onMouseButton);
    glfwSetCursorPosCallback(win,onCursor);
    glfwSetScrollCallback(win,onScroll);
    glfwSetKeyCallback(win,onKey);
    if(headless){
        autoRecord=true;                                  // headless implies recording
        if(autoRecFrames<=0) autoRecFrames=1800;          // default 30 s @ 60 fps
    }
    if(recFps<1) recFps=60;
    if(subSteps<1) subSteps=1;
    if(capW<16) capW=1920; if(capH<16) capH=1080;
    capW&=~1; capH&=~1;                       // x264 yuv420p needs even dimensions
    if(recCrf<0) recCrf=18; if(recCrf>51) recCrf=51;

    GLuint progStars = program(kStarVert,kStarFrag);
    GLuint progNebula= programNoise(kCloudVert,kNebulaFrag);
    GLuint progDust  = programNoise(kCloudVert,kDustFrag);
    GLuint progBright= program(kFsTriVert,kBrightFrag);
    GLuint progBlur  = program(kFsTriVert,kBlurFrag);
    GLuint progComp  = program(kFsTriVert,kCompositeFrag);

    GLuint fsVao; glGenVertexArrays(1,&fsVao);     // empty VAO for fullscreen passes

    // Per-preset particle budget. Big Bang runs at 1M with no nebulae (pure
    // gravitational structure formation); the others use the CLI star count
    // (default 100k) plus gas/dust fractions.
    auto countsFor=[&](int preset,int& ns,int& ng,int& nd){
        if(preset==7){ ns=(cliN>0?cliN:1000000); ng=0; nd=0; }
        else { ns=(cliN>0?cliN:100000);
               ng=(cliGas>=0?cliGas:ns/5); nd=(cliDust>=0?cliDust:ns/10); }
        if(ns<2) ns=2;
    };

    // Body buffers + CUDA interop. Rebuilt whenever the particle count changes
    // (e.g. switching to/from the 1M Big Bang preset), since the shared VBO is
    // fixed-size and registered with CUDA.
    int nStars=0,nGas=0,nDust=0,total=0;
    GLuint vboPos=0, starVao=0, vboStar=0, cloudVao=0, vboCloud=0;
    cudaGraphicsResource* cudaVbo=nullptr;
    auto buildBodyBuffers=[&](int ns,int ng,int nd){
        if(cudaVbo){ cudaGraphicsUnregisterResource(cudaVbo); cudaVbo=nullptr; }
        if(vboPos)   glDeleteBuffers(1,&vboPos);
        if(vboStar)  glDeleteBuffers(1,&vboStar);
        if(vboCloud) glDeleteBuffers(1,&vboCloud);
        if(starVao)  glDeleteVertexArrays(1,&starVao);
        if(cloudVao) glDeleteVertexArrays(1,&cloudVao);
        nStars=ns; nGas=ng; nDust=nd; total=ns+ng+nd;

        // shared dynamic VBO (CUDA writes x,y,z,speed for every body)
        glGenBuffers(1,&vboPos);
        glBindBuffer(GL_ARRAY_BUFFER,vboPos);
        glBufferData(GL_ARRAY_BUFFER,sizeof(float)*4*total,nullptr,GL_DYNAMIC_DRAW);

        // star VAO: pos (loc0) + intrinsic temperature/luminosity (loc1)
        glGenVertexArrays(1,&starVao); glGenBuffers(1,&vboStar);
        glBindVertexArray(starVao);
        glBindBuffer(GL_ARRAY_BUFFER,vboPos);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,sizeof(float)*4,(void*)0);
        {
            std::vector<float> star(2*nStars);
            std::mt19937 rng(99); std::uniform_real_distribution<float> uni(0,1);
            int bulge = nStars/200 + 1;
            for(int i=0;i<nStars;++i){
                float t, lum;
                if(i<bulge){ t=0.55f+0.25f*uni(rng); lum=1.6f+1.4f*uni(rng); }   // bright core
                else {
                    float u=uni(rng);
                    t = u*u*0.85f + 0.05f*uni(rng);          // mostly cool/red, few hot/blue
                    lum = 0.35f + 0.9f*t*t + 0.25f*uni(rng); // hotter stars brighter
                    if(uni(rng) > 0.985f) lum += 2.0f;       // rare bright giants
                }
                star[2*i+0]=t; star[2*i+1]=lum;
            }
            glBindBuffer(GL_ARRAY_BUFFER,vboStar);
            glBufferData(GL_ARRAY_BUFFER,sizeof(float)*2*nStars,star.data(),GL_STATIC_DRAW);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,(void*)0);
        }

        // cloud VAO: pos (loc0) + per-body (size,brightness) (loc1)
        glGenVertexArrays(1,&cloudVao); glGenBuffers(1,&vboCloud);
        glBindVertexArray(cloudVao);
        glBindBuffer(GL_ARRAY_BUFFER,vboPos);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,sizeof(float)*4,(void*)0);
        {
            std::vector<float> cattr;
            buildCloudAttrs(nStars,nGas,nDust,cattr);
            glBindBuffer(GL_ARRAY_BUFFER,vboCloud);
            glBufferData(GL_ARRAY_BUFFER,sizeof(float)*cattr.size(),
                         cattr.empty()?nullptr:cattr.data(),GL_STATIC_DRAW);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,(void*)0);
        }

        if(cudaGraphicsGLRegisterBuffer(&cudaVbo,vboPos,cudaGraphicsRegisterFlagsWriteDiscard)!=cudaSuccess)
            std::fprintf(stderr,"cudaGraphicsGLRegisterBuffer failed\n");
    };

    countsFor(p.preset,nStars,nGas,nDust);
    p.n=nStars; p.nGas=nGas; p.nDust=nDust;
    buildBodyBuffers(nStars,nGas,nDust);

    Simulation* sim = new Simulation(p);
    g_preset = p.preset;
    std::printf("Viewer: %d stars + %d gas + %d dust = %d bodies.\n"
        "  drag look | WASD move | Q/E down-up | Shift faster | scroll dolly | Space pause\n"
        "  V velocity | X auto-spin | N nebulae/dust | O orbit COM | C record | R reset cam | Esc quit\n"
        "  presets: 1 spiral  2 collision  3 minor-merger  4 collapse  5 explosion  6 head-on  7 BIG BANG(1M)\n",
        nStars,nGas,nDust,total);
    camReset();
    g_orbit = orbitOn;

    // HDR pipeline buffers, sized to the current render resolution.
    FBO scene, blurA, blurB, capture;       // capture = LDR target for video frames
    int curW=0, curH=0;                      // 0 -> (re)build on first frame

    // recording resolution (CLI --width/--height), independent of window size.
    const int CAP_W=capW, CAP_H=capH;
    FILE* ffmpeg=nullptr; int recW=0, recH=0;
    std::vector<unsigned char> frame;
    std::string lastFile;
    bool recording=false; int recFrames=0;

    double fpsT=glfwGetTime(); int frames=0;
    double recStartT=0, lastFrameT=glfwGetTime();   // headless progress timing
    bool orbitPrev=false; float com[3]={0,0,0};     // auto-orbit state
    if(autoRecord) g_toggleRecord=true;      // begin recording on the first frame

    while(!glfwWindowShouldClose(win)){
        // switch initial-condition template (keys 1..7) -> rebuild the simulation,
        // reallocating the body buffers first if the particle count changed.
        if(g_restart){
            g_restart=false;
            p.preset=g_preset;
            int ns,ng,nd; countsFor(p.preset,ns,ng,nd);
            if(ns!=nStars||ng!=nGas||nd!=nDust){ buildBodyBuffers(ns,ng,nd); camReset(); }
            p.n=nStars; p.nGas=nGas; p.nDust=nDust;
            delete sim; sim=new Simulation(p);   // re-seeds stars + gas + dust
            g_paused=false;
            std::printf("Preset %d loaded (%d bodies).\n", g_preset, total);
        }
        if(!g_paused) for(int s=0;s<subSteps;++s) sim->step();   // subSteps>1 -> faster evolution

        // Handle record start/stop first, so this frame renders at the right resolution.
        if(g_toggleRecord){
            g_toggleRecord=false;
            if(!recording){
                if(!ffmpegAvailable()){
                    std::printf("[REC] ffmpeg not found on PATH. Install it (winget install Gyan.FFmpeg).\n");
                } else {
                    recW=CAP_W; recH=CAP_H; frame.resize((size_t)recW*recH*4);
                    makeFBO8(capture,recW,recH);
                    // timestamped output file: gravity_capture_YYYYmmdd_HHMMSS.mp4
                    time_t now=time(nullptr); char ts[20];
                    std::strftime(ts,sizeof(ts),"%Y%m%d_%H%M%S",std::localtime(&now));
                    lastFile = "gravity_capture_" + std::string(ts) + ".mp4";
                    char cmd[640];
                    std::snprintf(cmd,sizeof(cmd),
                        "ffmpeg -y -loglevel error -f rawvideo -pixel_format rgba "
                        "-video_size %dx%d -framerate %d -i - -vf vflip -an "
                        "-c:v libx264 -preset slow -crf %d -pix_fmt yuv420p %s",
                        recW,recH,recFps,recCrf,lastFile.c_str());
                    ffmpeg=_popen(cmd,"wb");
                    if(ffmpeg){ recording=true; std::printf("[REC] recording %dx%d @%dfps -> %s\n",recW,recH,recFps,lastFile.c_str()); }
                    else std::printf("[REC] failed to launch ffmpeg.\n");
                }
            } else {
                recording=false;
                if(ffmpeg){ _pclose(ffmpeg); ffmpeg=nullptr; }
                std::printf("[REC] saved %s\n", lastFile.c_str());
            }
        }

        int fbw,fbh; glfwGetFramebufferSize(win,&fbw,&fbh);
        if(fbw<1) fbw=1; if(fbh<1) fbh=1;
        int rw = recording ? recW : fbw;         // render resolution
        int rh = recording ? recH : fbh;
        if(rw!=curW || rh!=curH){
            makeFBO(scene,rw,rh); makeFBO(blurA,rw/2,rh/2); makeFBO(blurB,rw/2,rh/2);
            curW=rw; curH=rh;
        }

        // CUDA writes positions into the shared VBO
        float* dptr=nullptr; size_t bytes=0;
        cudaGraphicsMapResources(1,&cudaVbo,0);
        cudaGraphicsResourceGetMappedPointer((void**)&dptr,&bytes,cudaVbo);
        sim->copyToRenderBuffer(dptr);
        cudaGraphicsUnmapResources(1,&cudaVbo,0);

        // --- free-fly camera: WASD move, Q/E down/up, Shift faster, drag to look ---
        float spd = 0.03f * (glfwGetKey(win,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ? 3.0f : 1.0f);
        auto moveCam=[&](const float*v,float s){ for(int i=0;i<3;++i) g_camPos[i]+=v[i]*s; };
        if(glfwGetKey(win,GLFW_KEY_W)==GLFW_PRESS) moveCam(g_f, spd);
        if(glfwGetKey(win,GLFW_KEY_S)==GLFW_PRESS) moveCam(g_f,-spd);
        if(glfwGetKey(win,GLFW_KEY_D)==GLFW_PRESS) moveCam(g_r, spd);
        if(glfwGetKey(win,GLFW_KEY_A)==GLFW_PRESS) moveCam(g_r,-spd);
        if(glfwGetKey(win,GLFW_KEY_E)==GLFW_PRESS) moveCam(g_u, spd);
        if(glfwGetKey(win,GLFW_KEY_Q)==GLFW_PRESS) moveCam(g_u,-spd);
        if(g_autospin){ vrot(g_f,g_u,0.0015f); vrot(g_r,g_u,0.0015f); }

        // --- auto-orbit the centre of mass (O / --orbit) ---
        sim->centerOfMass(com[0],com[1],com[2]);
        if(g_orbit){
            if(!orbitPrev){                                   // seed from current view (no jump)
                float dx=g_camPos[0]-com[0], dy=g_camPos[1]-com[1], dz=g_camPos[2]-com[2];
                g_orbR=std::sqrt(dx*dx+dy*dy+dz*dz); if(g_orbR<0.2f) g_orbR=0.2f;
                g_orbAz=std::atan2(dz,dx);
                g_orbEl=std::asin(std::max(-0.99f,std::min(0.99f,dy/g_orbR)));
            }
            g_orbAz += g_orbSpeed;
            float ce=std::cos(g_orbEl), se=std::sin(g_orbEl);
            g_camPos[0]=com[0]+g_orbR*std::cos(g_orbAz)*ce;
            g_camPos[1]=com[1]+g_orbR*se;
            g_camPos[2]=com[2]+g_orbR*std::sin(g_orbAz)*ce;
            float f[3]={com[0]-g_camPos[0],com[1]-g_camPos[1],com[2]-g_camPos[2]}; vnorm(f);
            float wup[3]={0,1,0}, r[3]; vcross(f,wup,r); vnorm(r);
            float u[3]; vcross(r,f,u);
            for(int i=0;i<3;++i){ g_f[i]=f[i]; g_r[i]=r[i]; g_u[i]=u[i]; }
        }
        orbitPrev=g_orbit;

        float camDist=std::sqrt(g_camPos[0]*g_camPos[0]+g_camPos[1]*g_camPos[1]+g_camPos[2]*g_camPos[2]);
        float ctr[3]={g_camPos[0]+g_f[0], g_camPos[1]+g_f[1], g_camPos[2]+g_f[2]};
        Mat4 proj=perspective(1.0f,(float)rw/(float)rh,0.02f,200.0f);
        Mat4 view=lookAt(g_camPos[0],g_camPos[1],g_camPos[2], ctr[0],ctr[1],ctr[2], g_u[0],g_u[1],g_u[2]);
        Mat4 mvp=mul(proj,view);

        // --- pass 1: stars into HDR scene buffer (additive) ---
        glBindFramebuffer(GL_FRAMEBUFFER,scene.fbo);
        glViewport(0,0,scene.w,scene.h);
        glClearColor(0.004f,0.004f,0.012f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glDisable(GL_DEPTH_TEST);

        // --- simulated nebulae + dust (realistic mode only) ---
        // The gas/dust are real tracer bodies in the shared buffer; we just draw
        // their index ranges. Drawn before the stars: dust is alpha-over so it
        // carves dark lanes into the glowing gas, and the additive stars shine
        // over the top.
        bool drawClouds = g_clouds && g_mode==0 && (nGas>0 || nDust>0);
        if(drawClouds){
            float cloudScale=(float)rh*1.83f;
            glBindVertexArray(cloudVao);
            if(nGas>0){                                  // emissive gas, additive
                glEnable(GL_BLEND); glBlendFunc(GL_ONE,GL_ONE);
                glUseProgram(progNebula);
                glUniformMatrix4fv(glGetUniformLocation(progNebula,"uMVP"),1,GL_FALSE,mvp.m);
                glUniform1f(glGetUniformLocation(progNebula,"uScale"),cloudScale);
                glDrawArrays(GL_POINTS,nStars,nGas);
            }
            if(nDust>0){                                 // dust, alpha-over (darkens)
                glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
                glUseProgram(progDust);
                glUniformMatrix4fv(glGetUniformLocation(progDust,"uMVP"),1,GL_FALSE,mvp.m);
                glUniform1f(glGetUniformLocation(progDust,"uScale"),cloudScale);
                glDrawArrays(GL_POINTS,nStars+nGas,nDust);
            }
        }

        // --- stars (additive) ---
        glEnable(GL_BLEND); glBlendFunc(GL_ONE,GL_ONE);
        glUseProgram(progStars);
        glUniformMatrix4fv(glGetUniformLocation(progStars,"uMVP"),1,GL_FALSE,mvp.m);
        glUniform1f(glGetUniformLocation(progStars,"uPointScale"),(float)rh*0.0022f*fmaxf(camDist,0.3f));
        glUniform1i(glGetUniformLocation(progStars,"uMode"),g_mode);
        glUniform1f(glGetUniformLocation(progStars,"uSpeedScale"),0.45f);
        glBindVertexArray(starVao);
        glDrawArrays(GL_POINTS,0,nStars);

        // --- bloom (realistic mode only): bright pass + separable blur ---
        bool doBloom = (g_mode == 0);
        if(doBloom){
            glDisable(GL_BLEND);
            glBindVertexArray(fsVao);
            glBindFramebuffer(GL_FRAMEBUFFER,blurA.fbo);
            glViewport(0,0,blurA.w,blurA.h);
            glUseProgram(progBright);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,scene.tex);
            glUniform1i(glGetUniformLocation(progBright,"uScene"),0);
            glUniform1f(glGetUniformLocation(progBright,"uThresh"),1.3f);
            glDrawArrays(GL_TRIANGLES,0,3);

            glUseProgram(progBlur);
            glUniform1i(glGetUniformLocation(progBlur,"uTex"),0);
            for(int it=0; it<2; ++it){
                glBindFramebuffer(GL_FRAMEBUFFER,blurB.fbo); glViewport(0,0,blurB.w,blurB.h);
                glBindTexture(GL_TEXTURE_2D,blurA.tex);
                glUniform2f(glGetUniformLocation(progBlur,"uDir"),1.0f/blurB.w,0.0f);
                glDrawArrays(GL_TRIANGLES,0,3);
                glBindFramebuffer(GL_FRAMEBUFFER,blurA.fbo); glViewport(0,0,blurA.w,blurA.h);
                glBindTexture(GL_TEXTURE_2D,blurB.tex);
                glUniform2f(glGetUniformLocation(progBlur,"uDir"),0.0f,1.0f/blurA.h);
                glDrawArrays(GL_TRIANGLES,0,3);
            }
        }

        // --- composite + tonemap; render into the capture buffer while recording ---
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_FRAMEBUFFER, recording ? capture.fbo : 0);
        glViewport(0,0,rw,rh);
        glUseProgram(progComp);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,scene.tex);
        glUniform1i(glGetUniformLocation(progComp,"uScene"),0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,blurA.tex);
        glUniform1i(glGetUniformLocation(progComp,"uBloom"),1);
        glUniform1f(glGetUniformLocation(progComp,"uExposure"),1.1f);
        glUniform1f(glGetUniformLocation(progComp,"uBloomAmt"), doBloom ? 0.55f : 0.0f);
        glBindVertexArray(fsVao);
        glDrawArrays(GL_TRIANGLES,0,3);

        // --- grab the frame for the video and blit a preview to the window ---
        if(recording && ffmpeg){
            glBindFramebuffer(GL_FRAMEBUFFER,capture.fbo);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glReadPixels(0,0,recW,recH,GL_RGBA,GL_UNSIGNED_BYTE,frame.data());
            std::fwrite(frame.data(),1,frame.size(),ffmpeg);
            ++recFrames;
            if(headless){                               // per-frame CLI progress
                double now=glfwGetTime();
                if(recStartT==0) recStartT=now;
                double inst = 1.0/((now-lastFrameT)>1e-6 ? now-lastFrameT : 1e-6);
                double avg  = recFrames/((now-recStartT)>1e-6 ? now-recStartT : 1.0);
                if(autoRecFrames>0){
                    double eta=(autoRecFrames-recFrames)/(avg>1e-6?avg:1e-6);
                    std::printf("\r[REC] frame %d/%d (%.1f%%) | %.1f fps x%d = %.0f steps/s | ETA %4.0fs   ",
                        recFrames,autoRecFrames,100.0*recFrames/autoRecFrames,
                        inst,subSteps,inst*subSteps,eta);
                } else {
                    std::printf("\r[REC] frame %d | %.1f fps x%d = %.0f steps/s   ",
                        recFrames,inst,subSteps,inst*subSteps);
                }
                std::fflush(stdout);
            }
            lastFrameT=glfwGetTime();
            if(autoRecFrames>0 && recFrames>=autoRecFrames) glfwSetWindowShouldClose(win,1);
            if(!headless){                              // mirror the frame to the window
                glBindFramebuffer(GL_READ_FRAMEBUFFER,capture.fbo);
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
                glBlitFramebuffer(0,0,recW,recH, 0,0,fbw,fbh, GL_COLOR_BUFFER_BIT, GL_LINEAR);
            }
        }

        if(!headless) glfwSwapBuffers(win);             // no buffer swap to a hidden window
        glfwPollEvents();

        if(++frames>=60){                               // window title fps (interactive only)
            double now=glfwGetTime(); double fps=frames/(now-fpsT);
            if(!headless){
                char title[220];
                std::snprintf(title,sizeof(title),
                    "CUDA Barnes-Hut 3D N-body | %d bodies | preset %d | %.1f FPS | %s%s%s",
                    total, g_preset, fps, g_mode? "velocity":"realistic",
                    g_paused?" | paused":"", recording?" | REC":"");
                glfwSetWindowTitle(win,title);
            }
            fpsT=now; frames=0;
        }
    }
    if(headless) std::printf("\n");

    if(ffmpeg) _pclose(ffmpeg);
    delete sim;
    glDeleteBuffers(1,&vboStar);  glDeleteVertexArrays(1,&starVao);
    glDeleteBuffers(1,&vboCloud); glDeleteVertexArrays(1,&cloudVao);
    cudaGraphicsUnregisterResource(cudaVbo);
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}

// ---------------------------------------------------------------------------
int main(int argc,char** argv){
    SimParams p; enum { VIEW, BENCH, VERIFY } mode=VIEW; int steps=200;
    bool autoRecord=false; int autoRecFrames=0;
    bool headless=false; int recFps=60, subSteps=1;
    int capW=1920, capH=1080, recCrf=18; bool orbitOn=false;
    int nArg=-1, gasArg=-1, dustArg=-1;        // <0 -> use per-preset default
    for(int i=1;i<argc;++i){ std::string a=argv[i];
        if(a=="--bench") mode=BENCH;
        else if(a=="--verify") mode=VERIFY;
        else if(a=="--record") autoRecord=true;
        else if(a=="--headless") headless=true;
        else if(a=="--orbit") orbitOn=true;
        else if(a=="--nogas"){ gasArg=0; dustArg=0; }
        else if(a.rfind("--recframes=",0)==0){ autoRecord=true; autoRecFrames=std::atoi(a.c_str()+12); }
        else if(a.rfind("--fps=",0)==0)   recFps=std::atoi(a.c_str()+6);
        else if(a.rfind("--substeps=",0)==0) subSteps=std::atoi(a.c_str()+11);
        else if(a.rfind("--width=",0)==0) capW=std::atoi(a.c_str()+8);
        else if(a.rfind("--height=",0)==0)capH=std::atoi(a.c_str()+9);
        else if(a.rfind("--crf=",0)==0)   recCrf=std::atoi(a.c_str()+6);
        else if(a.rfind("--n=",0)==0)     nArg=std::atoi(a.c_str()+4);
        else if(a.rfind("--gas=",0)==0)   gasArg=std::atoi(a.c_str()+6);
        else if(a.rfind("--dust=",0)==0)  dustArg=std::atoi(a.c_str()+7);
        else if(a.rfind("--preset=",0)==0)p.preset=std::atoi(a.c_str()+9);
        else if(a.rfind("--steps=",0)==0) steps=std::atoi(a.c_str()+8);
        else if(a.rfind("--theta=",0)==0) p.theta=(float)std::atof(a.c_str()+8);
        else if(a.rfind("--dt=",0)==0)    p.dt=(float)std::atof(a.c_str()+5);
        else { std::fprintf(stderr,"unknown arg: %s\n",a.c_str()); return 1; } }
    if(nArg>=0) p.n=nArg;
    if(p.n<2) p.n=2;
    // bench/verify stay pure-star (numbers comparable); the viewer picks the
    // star/gas/dust counts per preset (see countsFor in runViewer).
    switch(mode){ case BENCH: return runBench(p,steps); case VERIFY: return runVerify(p);
                  default: return runViewer(p,nArg,gasArg,dustArg,autoRecord,autoRecFrames,
                                            headless,recFps,subSteps,capW,capH,recCrf,orbitOn); }
}
