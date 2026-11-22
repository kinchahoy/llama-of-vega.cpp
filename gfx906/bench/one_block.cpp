#include <hip/hip_runtime.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <functional>
#define N 20000
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s line %d\n",hipGetErrorString(e),__LINE__);}}while(0)
typedef std::chrono::steady_clock C;
static double us(C::time_point a,C::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
double bench(std::function<void()> f){auto r=[&]{auto a=C::now();for(int i=0;i<N;i++)f();CK(hipDeviceSynchronize());return us(a,C::now())/N;};r();std::vector<double>v;for(int i=0;i<5;i++)v.push_back(r());return med(v);}
// A/B/C
__global__ void k_ew(float*d,const float*a,int per){size_t b=(size_t)blockIdx.x*per;for(int i=threadIdx.x;i<per;i+=blockDim.x)d[b+i]=a[b+i]*2.f+1.f;}
// D
__global__ void k_bar(float*d,const float*a,int E,int K){__shared__ float sh[1024];float v=0;for(int i=threadIdx.x;i<E;i+=blockDim.x){d[i]=a[i]*2.f+1.f;}
 for(int k=0;k<K;k++){sh[threadIdx.x]=v+k;__syncthreads();v=sh[(threadIdx.x+1)&1023];__syncthreads();} if(K)d[threadIdx.x]+=v;}
__device__ float wsum(float v){for(int o=32;o>0;o>>=1)v+=__shfl_xor(v,o,64);return v;}
__device__ float bred1(float v,float*sh){ // rms_norm_f32 style
 v=wsum(v);int w=threadIdx.x>>6,l=threadIdx.x&63;if(l==0)sh[w]=v;__syncthreads();
 v=l<(int)(blockDim.x>>6)?sh[l]:0.f;v=wsum(v);__syncthreads();return v;}
__device__ float bred2(float v,float*sh){sh[threadIdx.x]=v;__syncthreads();
 for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}float r=sh[0];__syncthreads();return r;}
template<int MODE> __global__ void k_row(float*d,const float*a,int E,int P){__shared__ float sh[1024];float sc=1.f;
 for(int p=0;p<P;p++){float s=0;for(int i=threadIdx.x;i<E;i+=blockDim.x){float x=a[i]*sc;s+=x*x;}
  s=MODE==1?bred1(s,sh):bred2(s,sh);sc=rsqrtf(s/E+1e-6f);}
 for(int i=threadIdx.x;i<E;i+=blockDim.x)d[i]=a[i]*sc;}
__global__ void k_g1(float*part,const float*a){__shared__ float sh[1024];float x=a[blockIdx.x*1024+threadIdx.x];float s=bred1(x*x,sh);if(threadIdx.x==0)part[blockIdx.x]=s;}
__global__ void k_g2(float*d,const float*a,const float*part,int nb){__shared__ float sh[1024];float v=threadIdx.x<nb?part[threadIdx.x]:0.f;float s=bred1(v,sh);float sc=rsqrtf(s/(nb*1024)+1e-6f);
 size_t i=(size_t)blockIdx.x*1024+threadIdx.x;d[i]=a[i]*sc;}
int main(){CK(hipSetDevice(0));size_t M=1<<20;float*A,*D,*Pt;CK(hipMalloc(&A,M*4));CK(hipMalloc(&D,M*4));CK(hipMalloc(&Pt,4096));hipMemset(A,0,M*4);hipMemset(D,0,M*4);
 int Es[]={1024,2304,4608,9216,18432,36864,73728};
 printf("A one block 1024thr (us)\n");for(int E:Es)printf(" E=%d %.2f\n",E,bench([&]{k_ew<<<1,1024>>>(D,A,E);}));
 printf("B split (per-block 1024 | 2304)\n");for(int E:Es){printf(" E=%d",E);for(int pb:{1024,2304}){if(E%pb){printf(" n/a");continue;}printf(" pb%d(g=%d) %.2f",pb,E/pb,bench([&]{k_ew<<<E/pb,1024>>>(D,A,pb);}));}printf("\n");}
 printf("C threads E=4608\n");for(int t:{64,128,256,512,1024})printf(" t=%d %.2f\n",t,bench([&]{k_ew<<<1,t>>>(D,A,4608);}));
 printf("D barrier-pairs K (2 syncs each) E=4608\n");for(int K:{0,1,2,4,8,16})printf(" K=%d %.2f\n",K,bench([&]{k_bar<<<1,1024>>>(D,A,4608,K);}));
 printf("E E1(shfl) E2(tree)\n");for(int E:{2304,4608,9216})printf(" E=%d E1 %.2f E2 %.2f\n",E,bench([&]{k_row<1><<<1,1024>>>(D,A,E,1);}),bench([&]{k_row<2><<<1,1024>>>(D,A,E,1);}));
 printf("F phases E=4608\n");for(int P:{1,2,3,4})printf(" P=%d %.2f\n",P,bench([&]{k_row<1><<<1,1024>>>(D,A,4608,P);}));
 printf("E1 extra: ");for(int E:{18432,36864,73728})printf(" E=%d %.2f",E,bench([&]{k_row<1><<<1,1024>>>(D,A,E,1);}));printf("\n");
 printf("G two-kernel vs E1\n");for(int E:{4608,9216,36864,73728}){int nb=(E+1023)/1024;
  double g=bench([&]{k_g1<<<nb,1024>>>(Pt,A);k_g2<<<nb,1024>>>(D,A,Pt,nb);});
  printf(" E=%d(nb=%d) G %.2f E1 %.2f\n",E,nb,g,bench([&]{k_row<1><<<1,1024>>>(D,A,E,1);}));}
}
