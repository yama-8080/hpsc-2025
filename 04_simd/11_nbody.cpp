#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 zeros = _mm512_setzero_ps();

  for(int i=0; i<N; i++) {
//    for(int j=0; j<N; j++) {
//      if(i != j) {
    //calc r
    __m512 ixvec = _mm512_set1_ps(x[i]);
    __m512 iyvec = _mm512_set1_ps(y[i]);
    __m512 rxvec = _mm512_sub_ps(ixvec, xvec);
    __m512 ryvec = _mm512_sub_ps(iyvec, yvec);
    __m512 rxvec2 = _mm512_mul_ps(rxvec, rxvec);
    __m512 ryvec2 = _mm512_mul_ps(ryvec, ryvec);
    __m512 sqrtvec = _mm512_add_ps(rxvec2, ryvec2);
    __m512 rvec = _mm512_sqrt_ps(sqrtvec);

    //set mask of the case i!=j
    __mmask16 mask = _mm512_cmp_ps_mask(rvec, zeros, _MM_CMPINT_NE);

    //calc upper and lower of the subtraction value from fx, fy
    __m512 xuppervec = _mm512_mul_ps(rxvec, mvec);
    __m512 yuppervec = _mm512_mul_ps(ryvec, mvec);
    __m512 rvec2 = _mm512_mul_ps(rvec, rvec);
    __m512 rvec3 = _mm512_mul_ps(rvec2, rvec);

    //calc subtraction value from fx, fy (reduction)
    __m512 subFxvec = _mm512_mask_div_ps(zeros, mask, xuppervec, rvec3);
    __m512 subFyvec = _mm512_mask_div_ps(zeros, mask, yuppervec, rvec3);
    float subFx = _mm512_reduce_add_ps(subFxvec);
    float subFy = _mm512_reduce_add_ps(subFyvec);

    //calc fx, fy
    fx[i] -= subFx;
    fy[i] -= subFy;
//      }
//    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
