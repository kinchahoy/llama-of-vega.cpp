#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s line %d\n",hipGetErrorString(e),__LINE__);}}while(0)
typedef std::chrono::steady_clock C;
static double us(C::time_point a,C::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
// MODE 0: add words, 1: int8->float fma with broadcast weights, 2: f16->float sum, 3: sdot4
template<int MODE> __device__ __forceinline__ float proc(uint4 v,const float*w,float acc){
  if(MODE==0){ return acc+(float)(v.x+v.y+v.z+v.w); }
  if(MODE==3){ int a=0; a=__builtin_amdgcn_sdot4(v.x,0x01020304,a,false); a=__builtin_amdgcn_sdot4(v.y,0x01020304,a,false);
    a=__builtin_amdgcn_sdot4(v.z,0x01020304,a,false); a=__builtin_amdgcn_sdot4(v.w,0x01020304,a,false); return acc+(float)a; }
  unsigned u[4]={v.x,v.y,v.z,v.w};
  if(MODE==1){
    #pragma unroll
    for(int i=0;i<4;i++){
      #pragma unroll
      for(int b=0;b<4;b++) acc=fmaf((float)(signed char)(u[i]>>(8*b)),w[b],acc);
    } return acc; }
  #pragma unroll
  for(int i=0;i<4;i++){ __half_raw lo,hi; lo.x=u[i]&0xffff; hi.x=u[i]>>16; acc+=(float)__half(lo)+(float)__half(hi);} return acc;
}
template<int T,int MODE,int PAT> __global__ __launch_bounds__(T) void k(const uint4*p,float*out,int nch){
  const uint4*b=p+(size_t)blockIdx.x*nch;
  float w[4]={0.5f,-1.f,2.f,0.25f}; float acc=0;
  if(PAT==0){
    #pragma unroll 4
    for(int c=threadIdx.x;c<nch;c+=T) acc=proc<MODE>(b[c],w,acc);
  } else {
    int per=(nch+T-1)/T; int s=threadIdx.x*per; int e=min(s+per,nch);
    #pragma unroll 4
    for(int c=s;c<e;c++) acc=proc<MODE>(b[c],w,acc);
  }
  for(int o=32;o>0;o>>=1) acc+=__shfl_xor(acc,o);
  __shared__ float sh[16];
  if((threadIdx.x&63)==0) sh[threadIdx.x>>6]=acc;
  __syncthreads();
  if(threadIdx.x==0){float s=0;for(int i=0;i<(T+63)/64;i++)s+=sh[i];out[blockIdx.x]=s;}
}
typedef void(*KF)(const uint4*,float*,int);
template<int T> KF pick(int mode,int pat){
  if(pat==0){ switch(mode){case 0:return k<T,0,0>;case 1:return k<T,1,0>;case 2:return k<T,2,0>;default:return k<T,3,0>;} }
  switch(mode){case 0:return k<T,0,1>;case 1:return k<T,1,1>;case 2:return k<T,2,1>;default:return k<T,3,1>;}
}
KF getk(int T,int mode,int pat){switch(T){case 64:return pick<64>(mode,pat);case 128:return pick<128>(mode,pat);case 256:return pick<256>(mode,pat);case 512:return pick<512>(mode,pat);default:return pick<1024>(mode,pat);}}
double LO,HI;
int main(int argc,char**argv){
  int dev=argc>1?atoi(argv[1]):0; int part=argc>2?atoi(argv[2]):0; // part 1: 256MB only
  CK(hipSetDevice(dev));
  size_t sizes[3]={3670016,33554432,268435456};
  uint4*buf;CK(hipMalloc(&buf,sizes[2]));CK(hipMemset(buf,1,sizes[2]));
  float*out;CK(hipMalloc(&out,8192*4*2));
  int Gs[]={64,128,256,512,1024,2048,8192}; int Ts[]={64,128,256,512,1024};
  auto timeit=[&](KF f,int G,int T,size_t S)->double{
    int nch=(int)(S/16/G); if(nch<1) return -1;
    auto L=[&](int n){for(int i=0;i<n;i++)hipLaunchKernelGGL(f,dim3(G),dim3(T),0,0,buf,out,nch);CK(hipStreamSynchronize(0));};
    L(2); auto a=C::now();L(3);double t=us(a,C::now())/3;
    int n=(int)std::min(20000.0,std::max(2.0,50000.0/t));
    std::vector<double> r;for(int j=0;j<5;j++){auto a=C::now();L(n);r.push_back(us(a,C::now())/n);}
    std::sort(r.begin(),r.end());LO=r[0];HI=r[4];return r[2];
  };
  auto line=[&](const char*tag,int mode,int pat,int G,int T,size_t S){
    double t=timeit(getk(T,mode,pat),G,T,S);
    if(t<0)return; printf("%s dev%d S=%zuMB mode=%d pat=%s G=%d T=%d us=%.2f GBs=%.1f min_us=%.2f max_us=%.2f\n",tag,dev,S>>20,mode,pat?"run":"chunk",G,T,t,S/t/1e3,LO,HI);fflush(stdout);};
  if(part==2){ for(int rep=0;rep<3;rep++){for(int G:{256,1024,2048,8192})for(int T:{256,1024})line("rep",0,0,G,T,sizes[1]); line("rep",0,0,8192,1024,sizes[2]); line("rep",0,0,512,128,sizes[0]);} return 0;}
  if(part==1){ for(int pat=0;pat<2;pat++)for(int G:Gs)for(int T:Ts)line("sweep",0,pat,G,T,sizes[2]); return 0;}
  for(int si=0;si<3;si++)for(int pat=0;pat<2;pat++)for(int G:Gs)for(int T:Ts)line("sweep",0,pat,G,T,sizes[si]);
  int G2[]={256,1024,8192},T2[]={128,256,1024};
  for(int mode:{1,3,2})for(int G:G2)for(int T:T2)line("mode",mode,0,G,T,sizes[1]);
  for(int mode:{1,3,2})for(int G:G2)for(int T:T2)line("mode",mode,0,G,T,sizes[2]);
}
