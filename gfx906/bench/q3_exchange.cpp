#include <hip/hip_runtime.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s line %d\n",hipGetErrorString(e),__LINE__);}}while(0)
typedef std::chrono::steady_clock C;
static double us(C::time_point a,C::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
__global__ void kw(float*p,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=i;}
__global__ void kr(const float*p,float*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&p[i]==-1.f)o[0]=1.f;}
__global__ void kt(float*o){if(threadIdx.x==0&&o[0]==-7.f)o[1]=1;}
struct R{double med,mn,mx;};
R meas(int reps,std::function<void()> f){for(int i=0;i<5;i++)f();std::vector<double>v;for(int i=0;i<reps;i++){auto a=C::now();f();v.push_back(us(a,C::now()));}
  std::sort(v.begin(),v.end());return {v[v.size()/2],v[0],v.back()};}
int main(){
  size_t P[]={2560,10240,20480,40960,163840,1<<20,16<<20}; size_t MAXP=16<<20;
  void *d[2],*hp[2],*hg[2]; hipStream_t s[2];
  for(int i=0;i<2;i++){CK(hipSetDevice(i));CK(hipMalloc(&d[i],MAXP));CK(hipMemset(d[i],0,MAXP));CK(hipHostMalloc(&hp[i],MAXP));hg[i]=malloc(MAXP);memset(hg[i],0,MAXP);CK(hipStreamCreate(&s[i]));}
  auto D2H=[&](int i,void*h,size_t n){CK(hipSetDevice(i));CK(hipMemcpyAsync(h,d[i],n,hipMemcpyDeviceToHost,s[i]));};
  auto H2D=[&](int i,void*h,size_t n){CK(hipSetDevice(i));CK(hipMemcpyAsync(d[i],h,n,hipMemcpyHostToDevice,s[i]));};
  auto syn=[&](int i){CK(hipSetDevice(i));CK(hipStreamSynchronize(s[i]));};
  auto duplex=[&](void**h,size_t n){D2H(0,h[0],n);D2H(1,h[1],n);syn(0);syn(1);H2D(0,h[1],n);H2D(1,h[0],n);syn(0);syn(1);};
  void*hpp[2]={hp[0],hp[1]};void*hgg[2]={hg[0],hg[1]};
  printf("single host thread with hipSetDevice before each call\n");
  printf("size_B | D2H0 | H2D1 | D2H1 | H2D0 | oneway0to1 | duplex | kernel+duplex+kernel | duplex_pageable | peerMemcpy (us, median; [min..max] if spread >10%%)\n");
  for(size_t p:P){int reps=p>=(16<<20)?20:200;
    std::vector<R> r;
    r.push_back(meas(reps,[&]{D2H(0,hp[0],p);syn(0);}));
    r.push_back(meas(reps,[&]{H2D(1,hp[1],p);syn(1);}));
    r.push_back(meas(reps,[&]{D2H(1,hp[1],p);syn(1);}));
    r.push_back(meas(reps,[&]{H2D(0,hp[0],p);syn(0);}));
    r.push_back(meas(reps,[&]{D2H(0,hp[0],p);syn(0);H2D(1,hp[0],p);syn(1);}));
    r.push_back(meas(reps,[&]{duplex(hpp,p);}));
    int n=p/4,bl=(n+255)/256;
    r.push_back(meas(reps,[&]{for(int i=0;i<2;i++){CK(hipSetDevice(i));hipLaunchKernelGGL(kw,dim3(bl),dim3(256),0,s[i],(float*)d[i],n);}syn(0);syn(1);
      duplex(hpp,p);
      for(int i=0;i<2;i++){CK(hipSetDevice(i));hipLaunchKernelGGL(kr,dim3(bl),dim3(256),0,s[i],(const float*)d[i],(float*)d[i],n);}syn(0);syn(1);}));
    r.push_back(meas(reps,[&]{duplex(hgg,p);}));
    r.push_back(meas(reps,[&]{CK(hipSetDevice(0));CK(hipMemcpyPeer(d[1],1,d[0],0,p));}));
    printf("%zu",p);for(auto&x:r){printf(" | %.1f",x.med);if(x.mx>1.1*x.mn&&(x.mx-x.mn)>0.1*x.med)printf("[%.1f..%.1f]",x.mn,x.mx);}printf("\n");fflush(stdout);
  }
  CK(hipSetDevice(0));
  R a=meas(2000,[&]{CK(hipStreamSynchronize(s[0]));});
  R b=meas(2000,[&]{hipLaunchKernelGGL(kt,dim3(1),dim3(64),0,s[0],(float*)d[0]);CK(hipStreamSynchronize(s[0]));});
  CK(hipSetDevice(1));
  R c=meas(2000,[&]{CK(hipStreamSynchronize(s[1]));});
  R e=meas(2000,[&]{hipLaunchKernelGGL(kt,dim3(1),dim3(64),0,s[1],(float*)d[1]);CK(hipStreamSynchronize(s[1]));});
  printf("sync idle dev0 %.2f (%.2f..%.2f) dev1 %.2f; kernel+sync dev0 %.2f (%.2f..%.2f) dev1 %.2f\n",a.med,a.mn,a.mx,c.med,b.med,b.mn,b.mx,e.med);
  R f=meas(2000,[&]{CK(hipSetDevice(0));CK(hipSetDevice(1));});
  printf("hipSetDevice pair %.2f us\n",f.med);
  int can01=-1,can10=-1;CK(hipDeviceCanAccessPeer(&can01,0,1));CK(hipDeviceCanAccessPeer(&can10,1,0));
  printf("canAccessPeer 0->1 %d 1->0 %d\n",can01,can10);
  for(size_t p:{(size_t)10240,(size_t)1<<20,(size_t)16<<20}){
    hipError_t e1=hipMemcpyPeer(d[1],1,d[0],0,p);
    int rp=p>=(16<<20)?20:200;
    R x=meas(rp,[&]{CK(hipSetDevice(0));CK(hipMemcpyPeer(d[1],1,d[0],0,p));});
    R y=meas(rp,[&]{CK(hipSetDevice(0));CK(hipMemcpyPeerAsync(d[1],1,d[0],0,p,s[0]));syn(0);});
    R z=meas(rp,[&]{CK(hipSetDevice(0));CK(hipMemcpyAsync(d[1],d[0],p,hipMemcpyDeviceToDevice,s[0]));syn(0);});
    printf("peer P=%zu rc=%d MemcpyPeer %.1f us, PeerAsync+sync %.1f us, DtoD-async %.1f us (%.2f GB/s)\n",p,(int)e1,x.med,y.med,z.med,p/x.med/1e3);
  }
}
