#include <hip/hip_runtime.h>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <functional>
#define N 20000
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s line %d\n",hipGetErrorString(e),__LINE__);}}while(0)
typedef std::chrono::steady_clock C;
static double us(C::time_point a,C::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
__global__ void k_triv(float*o){o[blockIdx.x*blockDim.x+threadIdx.x]=1.f;}
__global__ void k_fma(float*o){int i=blockIdx.x*blockDim.x+threadIdx.x; o[i]=o[i]*1.001f+0.5f;}
__global__ void k_a12(float*o,float*b,float*c,float*d,int a,int e,int f,int g,float*h,float*i2,int j,int k){o[threadIdx.x]=(float)(a+e+f+g+j+k);}
__global__ void k_dep(const float*in,float*o){int i=threadIdx.x; o[i]=in[i]+1.f;}
__global__ void k_fat(float*o,int K){int i=blockIdx.x*256+threadIdx.x; for(int s=0;s<K;s++){ o[i]=o[i]*1.001f+0.5f; __syncthreads();} }
double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
double bench(std::function<double()> f){f();std::vector<double>v;for(int i=0;i<5;i++)v.push_back(f());return med(v);}
float* buf(size_t n){float*p;CK(hipMalloc(&p,n*4));hipMemset(p,0,n*4);return p;}
int main(){
 for(int dev=0;dev<2;dev++){
  CK(hipSetDevice(dev)); printf("=== device %d\n",dev);
  size_t BIG=1024*256+4096; float*A=buf(BIG),*B=buf(BIG);
  hipStream_t s; CK(hipStreamCreate(&s));
  auto run1=[&](hipStream_t st,int g,int t,bool fma){return [=]()->double{auto a=C::now();for(int i=0;i<N;i++){ if(fma) k_fma<<<g,t,0,st>>>(A); else k_triv<<<g,t,0,st>>>(A);} CK(hipStreamSynchronize(st));return us(a,C::now())/N;};};
  printf("c1 default stream 1x256: %.2f\n",bench(run1(0,1,256,false)));
  printf("c1 created stream 1x256: %.2f\n",bench(run1(s,1,256,false)));
  {
   auto a=C::now();for(int i=0;i<N;i++)k_triv<<<1,256,0,s>>>(A);auto b=C::now();CK(hipStreamSynchronize(s));auto c=C::now();
   printf("c1 enqueue-only %.2f us/launch, total %.2f us/launch\n",us(a,b)/N,us(a,c)/N);}
  if(dev==0){
   for(int g:{1,9,64,256,1024}) printf("c2 grid %d x256 fma: %.2f\n",g,bench(run1(s,g,256,true)));
   for(int t:{64,256,1024}) printf("c3 1x%d: %.2f\n",t,bench(run1(s,1,t,false)));
   printf("c4 2args: %.2f\n",bench(run1(s,1,256,false)));
   printf("c4 12args: %.2f\n",bench([&]()->double{auto a=C::now();for(int i=0;i<N;i++)k_a12<<<1,256,0,s>>>(A,B,A,B,1,2,3,4,A,B,5,6);CK(hipStreamSynchronize(s));return us(a,C::now())/N;}));
  }
  for(int S:(dev==0?std::vector<int>{2,4,8}:std::vector<int>{4})){
   std::vector<hipStream_t> ss(S);std::vector<float*> bs(S);for(int i=0;i<S;i++){CK(hipStreamCreate(&ss[i]));bs[i]=buf(1024);}
   printf("c5 %d streams 1 thread: %.2f\n",S,bench([&]()->double{auto a=C::now();for(int i=0;i<N;i++)k_triv<<<1,256,0,ss[i%S]>>>(bs[i%S]);for(auto x:ss)CK(hipStreamSynchronize(x));return us(a,C::now())/N;}));
   printf("c5 %d streams %d threads: %.2f\n",S,S,bench([&]()->double{auto a=C::now();std::vector<std::thread>th;for(int t=0;t<S;t++)th.emplace_back([&,t]{CK(hipSetDevice(dev));for(int i=0;i<N/S;i++)k_triv<<<1,256,0,ss[t]>>>(bs[t]);CK(hipStreamSynchronize(ss[t]));});for(auto&x:th)x.join();return us(a,C::now())/N;}));
  }
  if(dev==1) continue;
  printf("c6 dependent chain: %.2f\n",bench([&]()->double{auto a=C::now();for(int i=0;i<N;i++){if(i&1)k_dep<<<1,256,0,s>>>(B,A);else k_dep<<<1,256,0,s>>>(A,B);}CK(hipStreamSynchronize(s));return us(a,C::now())/N;}));
  {float*bb[4];for(auto&x:bb)x=buf(1024);
  printf("c6 independent: %.2f\n",bench([&]()->double{auto a=C::now();for(int i=0;i<N;i++)k_dep<<<1,256,0,s>>>(A,bb[i&3]);CK(hipStreamSynchronize(s));return us(a,C::now())/N;}));}
  {
   hipGraph_t g;hipGraphExec_t ge;hipStream_t cs;CK(hipStreamCreate(&cs));
   hipError_t e=hipStreamBeginCapture(cs,hipStreamCaptureModeGlobal);
   if(e!=hipSuccess){printf("c7 capture unsupported: %s\n",hipGetErrorString(e));}
   else{for(int i=0;i<1000;i++)k_triv<<<1,256,0,cs>>>(A);
    e=hipStreamEndCapture(cs,&g); if(e==hipSuccess)e=hipGraphInstantiate(&ge,g,nullptr,nullptr,0);
    if(e!=hipSuccess)printf("c7 fail: %s\n",hipGetErrorString(e));
    else{printf("c7 graph 1000 chained nodes: %.3f us/node\n",bench([&]()->double{auto a=C::now();for(int i=0;i<20;i++)CK(hipGraphLaunch(ge,cs));CK(hipStreamSynchronize(cs));return us(a,C::now())/20000;}));}}
  }
  for(int K:{1,4,16,64}){int n=20000; printf("c8 fat K=%d: %.2f us/launch\n",K,bench([&]()->double{auto a=C::now();for(int i=0;i<n;i++)k_fat<<<9,256,0,s>>>(A,K);CK(hipStreamSynchronize(s));return us(a,C::now())/n;}));}
 }
}
