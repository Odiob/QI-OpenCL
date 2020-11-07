#pragma OPENCL EXTENSION cl_khr_fp64: enable


//2 component vector to hold the real and imaginary parts of a complex number:
typedef double2 cdouble;

#define I ((cdouble)(0.0, 1.0))


/*
 * Return Real (Imaginary) component of complex number:
 */
inline double  real(cdouble a){
     return a.x;
}
inline double  imag(cdouble a){
     return a.y;
}

inline cdouble cc(cdouble z){
	z.s1=-z.s1;
	return z;
}


/*
 * Multiply two complex numbers:
 *
 *  a = (aReal + I*aImag)
 *  b = (bReal + I*bImag)
 *  a * b = (aReal + I*aImag) * (bReal + I*bImag)
 *        = aReal*bReal +I*aReal*bImag +I*aImag*bReal +I^2*aImag*bImag
 *        = (aReal*bReal - aImag*bImag) + I*(aReal*bImag + aImag*bReal)
 */
inline cdouble  cmult(cdouble a, cdouble b){
    return (cdouble)( a.s0*b.s0 - a.s1*b.s1, a.s0*b.s1 + a.s1*b.s0);
}






__kernel void lanczos1(const double beta, __global cdouble *A, __global cdouble *v, __global cdouble *w, __global double *alpha)
{
	int id=get_global_id(0);
	int gid=get_group_id(0);
	int ls=get_local_size(0);
	int lid=get_local_id(0);
	int ngr=get_num_groups(0);	
	
	cdouble temp1,temp2;
	temp2=0;
	temp1=w[id]/beta;
	v[id]=temp1;

	int i;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(i=0;i<N;i++){
		temp2+=cmult(A[N*id+i],v[i]);
	}
	w[id]=temp2;

	alpha[id]=real(cmult(cc(temp2),temp1));
}




__kernel void lanczos2(const double alpha,const double beta, __global cdouble *v, __global cdouble *v2, __global cdouble *w, __global double *ab)
{
	
	int id=get_global_id(0);
	int gid=get_group_id(0);
	int ls=get_local_size(0);
	int lid=get_local_id(0);
	int ngr=get_num_groups(0);	
	
	cdouble temp1;
	temp1=w[id]-alpha*v[id]-beta*v2[id];
	w[id]=temp1;
	
	ab[id]=real(cmult(cc(temp1),temp1));
}




__kernel void transpose(__global double *A,__global double *At)
{
	int id=get_global_id(0);
	
	int i,j;
	i=id/N;
	j=id-i*N;
	At[j*N+i]=A[id];
}


__kernel void mat_mul(__global double *A,__global double *B,__global double *C)
{
	int id=get_global_id(0);
	
	int i,j,k;
	double temp=0;
	i=id/N;
	j=id-i*N;
	for(k=0;k<N;k++){
		temp+=A[i*N+k]*B[k*N+j];
	}
	C[id]=temp;
}

__kernel void transfer(__global double *A,__global double *diag,__global double *subdiag)
{
	int id=get_global_id(0);
	
	double temp,mu;
	mu=A[N*N-1];
	temp=A[id*N+id]-mu;
	diag[id]=temp;
	temp=A[id*N+id+1];
	subdiag[id]=temp;
	if(id==N-1)
		subdiag[id]=mu;
}