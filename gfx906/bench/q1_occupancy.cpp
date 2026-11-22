#include <hip/hip_runtime.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s line %d\n",hipGetErrorString(e),__LINE__);}}while(0)
typedef std::chrono::steady_clock C;
static double us(C::time_point a,C::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
template<int T,int N,int L,int M> __global__ __launch_bounds__(T,M) void k(float*out,long dt,float x,int*cnt){
  float a[N];
  #pragma unroll
  for(int i=0;i<N;i++) a[i]=threadIdx.x*0.001f+i;
  __shared__ float s[L?L:1];
  if(L) s[threadIdx.x%L]=a[0];
  if(threadIdx.x==0){int v=atomicAdd(&cnt[0],1)+1;atomicMax(&cnt[1],v);}
  long t0=wall_clock64();
  do{
    #pragma unroll
    for(int i=0;i<N;i++) a[i]=a[i]*x+a[(i+1)%N];
  }while(wall_clock64()-t0<dt);
  if(threadIdx.x==0)atomicSub(&cnt[0],1);
  float sum=0;
  #pragma unroll
  for(int i=0;i<N;i++) sum+=a[i];
  if(L){ __syncthreads(); sum+=s[(threadIdx.x+1)%L]; }
  if(sum==12345.678f) out[blockIdx.x]=sum;
}
struct V{const char*name;const void*fn;int T,N,L,M;};
std::vector<V> vs;
template<int T,int N,int L,int M> void add(){char*b=new char[64];snprintf(b,64,"T%d_N%d_L%dK_M%d",T,N,L*4/1024,M);vs.push_back({b,(const void*)k<T,N,L,M>,T,N,L*4,M});}
template<int T> void regs(){add<T,8,0,1>();add<T,32,0,1>();add<T,64,0,1>();add<T,100,0,1>();add<T,140,0,1>();add<T,200,0,1>();}
int main(int argc,char**argv){
  bool small=argc>1&&!strcmp(argv[1],"small");
  CK(hipSetDevice(0));
  regs<64>();regs<128>();regs<256>();regs<1024>();
  add<64,8,1024,1>();add<64,8,4096,1>();add<64,8,8192,1>();add<64,8,12288,1>();
  add<256,8,1024,1>();add<256,8,4096,1>();add<256,8,8192,1>();add<256,8,12288,1>();
  add<256,200,0,2>();add<256,200,0,4>();add<256,200,0,8>();
  add<128,200,0,2>();add<128,200,0,4>();add<128,200,0,8>();
  add<64,200,0,2>();add<64,200,0,4>();add<64,200,0,8>();
  int rate=0;CK(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0)); // kHz
  double Dus=argc>2?atof(argv[2]):30.0; long dt=(long)(Dus*rate/1000.0);
  float*o;CK(hipMalloc(&o,1<<20));int*cnt;CK(hipMalloc(&cnt,8));
  printf("wallclock kHz %d, D=%.0fus = %ld ticks\n",rate,Dus,dt);
  std::vector<int> Gs={16,64,128,192,256,384,512,768,1024,1536,2048,3072,4096};
  if(small)Gs={16,64};
  printf("variant T N LDSbytes M | numRegs localB sharedB apiOcc | us(G):");
  for(int g:Gs)printf(" %d",g);printf(" | Rest_per_CU\n");
  hipStream_t st;CK(hipStreamCreate(&st));
  for(auto&v:vs){
    hipFuncAttributes at;CK(hipFuncGetAttributes(&at,v.fn));
    int occ=-1;CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ,v.fn,v.T,0));
    std::vector<double> t;
    int rmax=0;{CK(hipMemset(cnt,0,8));float xx=0.9999f;void*args[]={&o,&dt,&xx,&cnt};hipLaunchKernel(v.fn,dim3(4096),dim3(v.T),args,0,st);CK(hipStreamSynchronize(st));int h[2];CK(hipMemcpy(h,cnt,8,hipMemcpyDeviceToHost));rmax=h[1];}
    for(int g:Gs){
      auto launch=[&](int n){for(int i=0;i<n;i++){float xx=0.9999f;void*args[]={&o,&dt,&xx,&cnt};hipLaunchKernel(v.fn,dim3(g),dim3(v.T),args,0,st);}CK(hipStreamSynchronize(st));};
      launch(2);
      std::vector<double> r;
      for(int j=0;j<(small?1:5);j++){auto a=C::now();launch(10);r.push_back(us(a,C::now())/10);}
      std::sort(r.begin(),r.end());t.push_back(r[r.size()/2]);
    }
    printf("%s %d %d %d %d | %d %zu %zu %d |",v.name,v.T,v.N,v.L,v.M,at.numRegs,at.localSizeBytes,at.sharedSizeBytes,occ);
    for(double x:t)printf(" %.0f",x);
    double ratio=t.back()/t[0]; double rounds=std::max(1.0,ratio);
    printf(" | ratio %.2f Rest/CU %.2f | Rmax_counter %d = %.2f/CU\n",ratio,Gs.back()/rounds/64.0,rmax,rmax/64.0);
    fflush(stdout);
  }
}
