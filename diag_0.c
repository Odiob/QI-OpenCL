// uses Lanczos algorithm to tridiagonalize a random hermitian matrix and then the divide & conquer algorithm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <string.h>
#include <lapacke.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)


int N;	//dimension of the hermitian matrix





//PROTOTYPES

void init(double complex *cmat,double *mat);
void prtcmpx(double complex z);
double complex dot(double complex *u, double complex *v);
double norm(double complex *u);
double complex* matvec(double complex *M, double complex *v);
double* QR(double *diag,double *subdiag);
double* matmul(double *A, double *B);
double* transp(double *A);





//MAIN

int main(){
	
	N=514; //only even numbers
	char str[40];
	sprintf(str,"-cl-std=CL1.2 -D N=%d",N);
	FILE *f;
	int i,j,k;
	double *M;
	double complex *cM;
	M=(double*)malloc(sizeof(double)*2*N*N);
	cM=malloc(sizeof(double complex)*N*N);
	init(cM,M);
	
	
	
	//opencl initialization
	
	
	int dev=1;
	cl_platform_id *platforms;
	cl_uint num_platforms;
    cl_device_id device;   
    cl_uint num_devices;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	size_t glob_size;
	size_t glob_size2;
	size_t loc_size=N;
    
	clGetPlatformIDs(5,NULL,&num_platforms);
	platforms=(cl_platform_id*) malloc(sizeof(cl_platform_id)*num_platforms);
	clGetPlatformIDs(num_platforms,platforms,NULL);
	printf("Number of Platforms detected: %d\n",num_platforms);
	char name[40];
	for(i=0;i<num_platforms;i++){
		clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,sizeof(name),&name,NULL);
		printf("%s\n",name);
	}
	
	
	if(dev==1){
		clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU,1,&device,&num_devices);
		for(i=1;loc_size>256;i++){
			if(loc_size%i==0)
				loc_size=N/i;
		}
		printf("\nAMD GPU selected\n");
	}
	else if(dev==2){
		clGetDeviceIDs(platforms[1],CL_DEVICE_TYPE_CPU,1,&device,&num_devices);
		for(i=1;loc_size>8192;i++){
			if(loc_size%i==0)
				loc_size=N/i;
		}
		printf("\nIntel CPU selected\n");
	}
	
	context=clCreateContext( NULL, 1,&device, NULL, NULL, NULL);
	printf("\nContext created\n");
	
	queue=clCreateCommandQueueWithProperties(context,device,NULL,NULL);
	printf("\nCommand queue created\n");
	
	char *source_str;
    size_t source_size;
    f=fopen("kernels.cl", "r");
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, f);
    fclose(f);
	program=clCreateProgramWithSource(context,1,(const char **)&source_str,(const size_t *)&source_size,NULL);
	clBuildProgram(program,1,&device,str,NULL,NULL);
	printf("\nProgram created and built\n");
	free(source_str);
	
	cl_kernel lanczos1 = clCreateKernel(program, "lanczos1", NULL);
	printf("\nKernel \"Lanczos1\" created\n");
	
	cl_kernel lanczos2 = clCreateKernel(program, "lanczos2", NULL);
	printf("\nKernel \"Lanczos2\" created\n");
	
	cl_kernel transpose = clCreateKernel(program, "transpose", NULL);
	printf("\nKernel \"Transpose\" created\n");
	
	cl_kernel mat_mul = clCreateKernel(program, "mat_mul", NULL);
	printf("\nKernel \"Mat_mul\" created\n");
	
	cl_kernel transfer = clCreateKernel(program, "transfer", NULL);
	printf("\nKernel \"Transfer\" created\n");
	
	
	
	
	//lanczos algorithm
	
	
	double a,b=1;
	double *P, *v, *vold,*w;
	double *diag,*subdiag,*ab;
	
	cl_double alpha;
	cl_double beta;
	cl_mem M_buff;
	cl_mem v_buff;
	cl_mem vold_buff;
	cl_mem w_buff;
	cl_mem ab_buff;
	
	diag=malloc(sizeof(double)*N);
	subdiag=malloc(sizeof(double)*N);
	w=calloc(2*N,sizeof(double));
	v=malloc(sizeof(double)*2*N);
	vold=calloc(2*N,sizeof(double));
	P=calloc(2*N*N,sizeof(double));
	ab=malloc(N*sizeof(double));
	P[0]=1;
	P[1]=0;
	w[0]=P[0];
	w[1]=P[1];
	
	glob_size=N;
	M_buff=clCreateBuffer(context,CL_MEM_READ_ONLY,2*N*N*sizeof(double),NULL,NULL);
	v_buff=clCreateBuffer(context,CL_MEM_READ_WRITE,2*N*sizeof(double),NULL,NULL);
	vold_buff=clCreateBuffer(context,CL_MEM_READ_WRITE,2*N*sizeof(double),NULL,NULL);
	w_buff=clCreateBuffer(context,CL_MEM_READ_WRITE,2*N*sizeof(double),NULL,NULL);
	ab_buff=clCreateBuffer(context,CL_MEM_WRITE_ONLY,N*sizeof(double),NULL,NULL);
	
	for(i=0;i<N;i++){
		
		beta=b;
		
		clEnqueueWriteBuffer(queue, M_buff, CL_TRUE, 0,2*N*N*sizeof(double), M, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, w_buff, CL_TRUE, 0,2*N*sizeof(double), w, 0, NULL, NULL);
		clSetKernelArg(lanczos1, 0, sizeof(cl_double), &beta);
		clSetKernelArg(lanczos1, 1, sizeof(cl_mem), &M_buff);
		clSetKernelArg(lanczos1, 2, sizeof(cl_mem), &v_buff);
		clSetKernelArg(lanczos1, 3, sizeof(cl_mem), &w_buff);
		clSetKernelArg(lanczos1, 4, sizeof(cl_mem), &ab_buff);
		clEnqueueNDRangeKernel(queue,lanczos1,1,NULL,&glob_size,&loc_size,0,NULL,NULL);
		clEnqueueReadBuffer(queue,v_buff, CL_TRUE, 0,2*N*sizeof(double), v, 0, NULL, NULL);
		clEnqueueReadBuffer(queue,ab_buff, CL_TRUE, 0,N*sizeof(double), ab, 0, NULL, NULL);
		
		a=ab[0];
		for(j=1;j<N;j++){
			a+=ab[j];
		}
		diag[i]=a;
		
		if(i==0)
			beta=0;
		
		alpha=a;
		
		clEnqueueWriteBuffer(queue, vold_buff, CL_TRUE, 0,2*N*sizeof(double), vold, 0, NULL, NULL);
		clSetKernelArg(lanczos2, 0, sizeof(cl_double), &alpha);
		clSetKernelArg(lanczos2, 1, sizeof(cl_double), &beta);
		clSetKernelArg(lanczos2, 2, sizeof(cl_mem), &v_buff);
		clSetKernelArg(lanczos2, 3, sizeof(cl_mem), &vold_buff);
		clSetKernelArg(lanczos2, 4, sizeof(cl_mem), &w_buff);
		clSetKernelArg(lanczos2, 5, sizeof(cl_mem), &ab_buff);
		clEnqueueNDRangeKernel(queue,lanczos2,1,NULL,&glob_size,&loc_size,0,NULL,NULL);
		clEnqueueReadBuffer(queue,w_buff, CL_TRUE, 0,2*N*sizeof(double), w, 0, NULL, NULL);
		clEnqueueReadBuffer(queue,ab_buff, CL_TRUE, 0,N*sizeof(double), ab, 0, NULL, NULL);
		
		b=ab[0];
		for(j=1;j<N;j++){
			b+=ab[j];
		}
		b=sqrt(b);
		subdiag[i]=b;
		
		for(j=0;j<N;j++){
			P[2*j*N+2*i]=v[2*j];
			P[2*j*N+2*i+1]=v[2*j+1];
			vold[2*j]=v[2*j];
			vold[2*j+1]=v[2*j+1];
		}
	}
	
	clReleaseMemObject(M_buff);
	clReleaseMemObject(v_buff);
	clReleaseMemObject(vold_buff);
	clReleaseMemObject(w_buff);
	clReleaseMemObject(ab_buff);
	clReleaseKernel(lanczos1);
	clReleaseKernel(lanczos2);
	
	free(w);
	free(v);
	free(vold);
	
	
	
	//LAPACK
	
	lapack_int n;
	n=N;
	double *eig;
	eig=malloc(sizeof(double)*N);

	LAPACKE_zheev(LAPACK_ROW_MAJOR,'V','U',n,cM,n,eig);
	
	f=fopen("lap.txt","w");
	for(i=0;i<N;i++){
		fprintf(f,"%.10f\n",eig[i]);
	}
	fprintf(f,"\n\n\n\n\n");
	for(i=0;i<N;i++){
		for(j=0;j<N;j++)
			fprintf(f,"(%f, %f)	",creal(M[i*N+j]),cimag(M[i*N+j]));
		fprintf(f,"\n");
	}
	fclose(f);
	
	free(eig);
	free(M);
	free(cM);
	
	
	
	//QR algorithm
	
	int chk=0;
	double mu,conf,conf2=0;
	double *Q,*A;
	
	cl_mem A_buff;
	cl_mem B_buff;
	cl_mem C_buff;
	
	glob_size=N*N;
	glob_size2=N;
	A_buff=clCreateBuffer(context,CL_MEM_READ_WRITE,N*N*sizeof(double),NULL,NULL);
	B_buff=clCreateBuffer(context,CL_MEM_READ_WRITE,N*N*sizeof(double),NULL,NULL);
	C_buff=clCreateBuffer(context,CL_MEM_READ_WRITE,N*N*sizeof(double),NULL,NULL);
	v_buff=clCreateBuffer(context,CL_MEM_WRITE_ONLY,N*sizeof(double),NULL,NULL);
	w_buff=clCreateBuffer(context,CL_MEM_WRITE_ONLY,N*sizeof(double),NULL,NULL);
	
	mu=diag[N-1];
	A=malloc(sizeof(double)*N*N);
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			if(i==j){
				A[i*N+j]=diag[i];
			}
			else if(i==j+1){
				A[i*N+j]=subdiag[i-1];
			}
			else if(i==j-1){
				A[i*N+j]=subdiag[i];
			}
			else{
				A[i*N+j]=0;
			}
		}
		diag[i]-=mu;
	}
	
	Q=QR(diag,subdiag);
	while(chk==0){
		double *tdiag,*tsubdiag,*Qnew;
		
		//temp=A*Q
		clEnqueueWriteBuffer(queue, A_buff, CL_TRUE, 0,N*N*sizeof(double), A, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, B_buff, CL_TRUE, 0,N*N*sizeof(double), Q, 0, NULL, NULL);
		clSetKernelArg(mat_mul, 0, sizeof(cl_mem), &A_buff);
		clSetKernelArg(mat_mul, 1, sizeof(cl_mem), &B_buff);
		clSetKernelArg(mat_mul, 2, sizeof(cl_mem), &C_buff);
		clEnqueueNDRangeKernel(queue,mat_mul,1,NULL,&glob_size,NULL,0,NULL,NULL);
		
		//Qt=trasp(Q)
		clEnqueueWriteBuffer(queue, B_buff, CL_TRUE, 0,N*N*sizeof(double), Q, 0, NULL, NULL);
		clSetKernelArg(transpose, 0, sizeof(cl_mem), &B_buff);
		clSetKernelArg(transpose, 1, sizeof(cl_mem), &A_buff);
		clEnqueueNDRangeKernel(queue,transpose,1,NULL,&glob_size,NULL,0,NULL,NULL);
		
		//Ak=Qt*temp
		clSetKernelArg(mat_mul, 0, sizeof(cl_mem), &A_buff);
		clSetKernelArg(mat_mul, 1, sizeof(cl_mem), &C_buff);
		clSetKernelArg(mat_mul, 2, sizeof(cl_mem), &B_buff);
		clEnqueueNDRangeKernel(queue,mat_mul,1,NULL,&glob_size,NULL,0,NULL,NULL);
		
		tdiag=malloc(sizeof(double)*N);
		tsubdiag=malloc(sizeof(double)*N);
		
		clSetKernelArg(transfer, 0, sizeof(cl_mem), &B_buff);
		clSetKernelArg(transfer, 1, sizeof(cl_mem), &v_buff);
		clSetKernelArg(transfer, 2, sizeof(cl_mem), &w_buff);
		clEnqueueNDRangeKernel(queue,transfer,1,NULL,&glob_size2,NULL,0,NULL,NULL);
		clEnqueueReadBuffer(queue,v_buff, CL_TRUE, 0,N*sizeof(double), tdiag, 0, NULL, NULL);
		clEnqueueReadBuffer(queue,w_buff, CL_TRUE, 0,N*sizeof(double), tsubdiag, 0, NULL, NULL);
		
		mu=tsubdiag[N-1];
		tdiag[N-1]=0;
		Qnew=QR(tdiag,tsubdiag);
		
		//Q=Q*Qnew
		clEnqueueWriteBuffer(queue, A_buff, CL_TRUE, 0,N*N*sizeof(double), Q, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, B_buff, CL_TRUE, 0,N*N*sizeof(double), Qnew, 0, NULL, NULL);
		clSetKernelArg(mat_mul, 0, sizeof(cl_mem), &A_buff);
		clSetKernelArg(mat_mul, 1, sizeof(cl_mem), &B_buff);
		clSetKernelArg(mat_mul, 2, sizeof(cl_mem), &C_buff);
		clEnqueueNDRangeKernel(queue,mat_mul,1,NULL,&glob_size,NULL,0,NULL,NULL);
		clEnqueueReadBuffer(queue,C_buff, CL_TRUE, 0,N*N*sizeof(double), Q, 0, NULL, NULL);
		free(Qnew);

		conf=tdiag[0];
		for(i=0;i<N;i++){
			if(tdiag[i]<conf)
				conf=tdiag[i];
		}
		free(tdiag);
		free(tsubdiag);
		conf+=mu;
		if(fabs(conf-conf2)< 1e-13)
			chk=1;
		conf2=conf;
		printf("%f\n",conf2);
	}
	
	printf("%.10f",conf2);
	

	
	
	
	//free memory
	free(P);
	free(diag);
	free(subdiag);
	free(A);
	free(Q);
	
	//Deallocate resources */
	clReleaseMemObject(A_buff);
	clReleaseMemObject(B_buff);
	clReleaseMemObject(C_buff);
	clReleaseKernel(mat_mul);
	clReleaseKernel(transpose);
	clReleaseKernel(transfer);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
		
	return 0;
}





//FUNCTIONS


//random hermitian matrix generation

void init(double complex *cmat,double *mat){
	double x,y;
	int i,j;
	srand((unsigned int)time(NULL));
	for(i=0;i<N;i++){
		for(j=i;j<N;j++){
			if(i==j){
				x = ((double)rand()/(double)(RAND_MAX)) * 200-100;
				y=0;
				cmat[i*N+j]=x+y*I;
				mat[i*2*N+2*j]=x;
				mat[i*2*N+2*j+1]=y;
			}
			else{
				x = ((double)rand()/(double)(RAND_MAX)) * 200-100;
				y = ((double)rand()/(double)(RAND_MAX)) * 200-100;
				cmat[i*N+j]=x+y*I;
				cmat[j*N+i]=x-y*I;
				mat[i*2*N+2*j]=x;
				mat[i*2*N+2*j+1]=y;
				mat[j*2*N+2*i]=x;
				mat[j*2*N+2*i+1]=-y;
			}
		}
	}
	return;
}


//print complex number

void prtcmpx(double complex z){
	printf("(%f,%f)\n",creal(z),cimag(z));
	return;
}


//dot product

double complex dot(double complex *u, double complex *v){
	int i;
	double complex z=0+0*I;
	for(i=0;i<N;i++)
		z+=conj(u[i])*v[i];
	return z;
}


//euclidean norm

double norm(double complex *u){
	return sqrt(cabs(dot(u,u)));
}


//QR decomposition

double* QR(double *diag,double *subdiag){
	
	int i,j;
	double norm,gam,sig,temp1,temp2,temp3;
	double *Q;
	
	Q=(double*)calloc(N*N,sizeof(double));
	norm=sqrt(diag[0]*diag[0]+subdiag[0]*subdiag[0]);
	gam=diag[0]/norm;
	sig=subdiag[0]/norm;
	Q[0]=gam;
	Q[1]=-sig;
	Q[N]=sig;
	Q[N+1]=gam;
	temp1=-sig*subdiag[0]+gam*diag[1];
	temp2=gam*subdiag[1];

	for(i=1;i<N-1;i++){
		norm=sqrt(temp1*temp1+subdiag[i]*subdiag[i]);
		gam=temp1/norm;
		sig=subdiag[i]/norm;
		for(j=0;j<i+1;j++){
			temp3=Q[j*N+i];
			Q[j*N+i]=temp3*gam;
			Q[j*N+i+1]=-temp3*sig;
		}
		Q[(i+1)*N+i]=sig;
		Q[(i+1)*N+i+1]=gam;
		temp1=-sig*temp2+gam*diag[i+1];
		temp2=gam*subdiag[i+1];
	}
	return Q;
}

