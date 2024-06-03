#include "mex.h"
#include "matrix.h"
#include "math.h"
//==========================================================================================================================
//==========================================================================================================================
int m_w = 521288629;    /* must not be zero */
int m_z = 362436069;    /* must not be zero */

static int GetUint()
{
    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    return (m_z << 16) + m_w;
}

static double GetUniform()
{
    // 0 <= u < 2^32
    int u = GetUint();
    // The magic number below is 1/(2^32 + 2).
    // The result is strictly between 0 and 1.
    return (u + 1.0) * 2.328306435454494e-10+0.5;
}
//==========================================================================================================================
//==========================================================================================================================
double varDt(int m_A, double r[m_A], int ir[m_A], double d[m_A][3], int edg[m_A][2], 
            mwSize m_coord, double f_mod[m_coord][3], double dr,  double mu, int local,
            mwSize m_fn, int in[m_fn], mwSize m_rg, double rg[m_rg], int iEdgExo, bool restored) {
//-------------------------------------------------------------------------
    double f1[m_A][3];
    double f2[m_A][3];
    for (int m=0; m<m_A; m++){
        f1[m][0]=f_mod[edg[m][0]][0]; f1[m][1]=f_mod[edg[m][0]][1]; f1[m][2]=f_mod[edg[m][0]][2];
        f2[m][0]=f_mod[edg[m][1]][0]; f2[m][1]=f_mod[edg[m][1]][1]; f2[m][2]=f_mod[edg[m][1]][2];
    }
    double df[m_A][3];
    for (int m=0; m<m_A; m++){
        df[m][0]=(f2[m][0]-f1[m][0]); df[m][1]=(f2[m][1]-f1[m][1]); df[m][2]=(f2[m][2]-f1[m][2]);
    }
    double SorE[m_A];  //-:shrink +:extend
    bool iSorE[m_A];   //Shrink: true  Extend: false 
    for (int m=0; m<m_A; m++){
        SorE[m]=df[m][0]*d[m][0]+df[m][1]*d[m][1]+df[m][2]*d[m][2];  
        if (SorE[m]<0){
            iSorE[m]=true;
        }
        else {
            iSorE[m]=false;
        }
    }
    double c_1[m_A],c_2[m_A],b[m_A],Delta[m_A],a[m_A],c[m_A];
    bool idWrong[m_A];
    int idMin=10000, idMax=0;
    int irMin=100000000,irMax=0;
    double rMin=100000, rMax=0;
    for (int m=0; m<m_A; m++){
        r[m]=floor(r[m]/dr+0.5)*dr;
    }
    if (restored==true){
        for (int m=0; m<m_A; m++){
        if (ir[m]<irMin){
            irMin=ir[m];}
        if (ir[m]>irMax){
            irMax=ir[m];}
        if (r[m]<rMin){
            rMin=r[m];}
        if (r[m]>rMax){
            rMax=r[m];}
        }
        mexPrintf("%d , %d ; %f , %f ; ",irMin,irMax,rMin,rMax);
        for (int m=0; m<m_A; m++){
            if (in[ir[m]]<idMin) {
                idMin=in[ir[m]];}
            if (in[ir[m]]>idMax) {
                idMax=in[ir[m]];}
        }
        mexPrintf(" %d , %d \n",idMin,idMax);
    }
    for (int m=0; m<m_A; m++){
        //c_1:shrink, c_2: extend
        c_1[m]=r[m]*r[m]-pow((0.5*(rg[in[ir[m]]-2]+rg[in[ir[m]]-1])),2);
        c_2[m]=r[m]*r[m]-pow((0.5*(rg[in[ir[m]]]+rg[in[ir[m]]+1])),2);
        b[m]=2*mu*((d[m][0])*(f_mod[edg[m][1]][0]-f_mod[edg[m][0]][0])
                  +(d[m][1])*(f_mod[edg[m][1]][1]-f_mod[edg[m][0]][1])
                  +(d[m][2])*(f_mod[edg[m][1]][2]-f_mod[edg[m][0]][2]));
//         c[m]=c_1[m];
//         if (iSorE[m]==false) {
//             c[m]=c_2[m];
//         }
//         a[m]=mu*mu*(df[m][0]*df[m][0]+df[m][1]*df[m][1]+df[m][2]*df[m][2]);
//         Delta[m]=b[m]*b[m]-4*a[m]*c[m];
//         if (Delta[m]<0){
//         idWrong[m]=true;}
//         else {
//         idWrong[m]=false;}
    }
//     mexPrintf(" s4\n");
//     for (int m=0; m<m_A; m++){
//         if (idWrong[m]==true){
//             c[m]=c_2[m];
//             if (iSorE[m]==false) {
//               c[m]=c_1[m];
//             }
//             Delta[m]=b[m]*b[m]-4*a[m]*c[m];
//         }
//     }
    double dtMax=1e15;
    double dt=dtMax;
    double dtTem[2];
    double dtTemMin;
    for (int m=0; m<m_A; m++){
//         if (Delta[m]<0){
//             mexPrintf(" Delta wrong! %f\n",Delta[m]);
//         }
//         dtTem[1]=(-b[m]+sqrt(Delta[m]))/(2.*a[m]);
//         dtTem[2]=(-b[m]-sqrt(Delta[m]))/(2.*a[m]);
        dtTem[1]=c_1[m]/b[m];
        dtTem[2]=c_2[m]/b[m];
        if (dtTem[1]<0) {
            dtTem[1]=dtMax;
        }
        if (dtTem[2]<0) {
            dtTem[2]=dtMax;
        }
        dtTemMin=dtTem[1];
        if (dtTemMin>dtTem[2]) {
            dtTemMin=dtTem[2];
        }
        if (dtTemMin<dt) {
            dt=dtTemMin;}
//         if ((dtTemMin<dt)&&(local==1)) {
//             dt=dtTemMin;}
//         else if ((dtTemMin<dt)&&(local>1)){
//             if (iEdgExo!=m) {
//             dt=dtTemMin;}
//         }
    }
//-------------------------------------------------------------------------    
   return dt;
}
//==========================================================================================================================
//==========================================================================================================================
void arrayComp (double* coord_in, double* id_on_coord_in, double* A_in, double* pm_in,double* fn,
                double* RLevy,double* in_in,double* rg,
                double* outM, double* outM2, double* outM3, double* outM4,
                mwSize m_coord, mwSize m_id_on_coord, mwSize m_A_in, mwSize m_fn, mwSize m_RLevy, mwSize m_rg)
{
//----------------------------------------------------------------------------------------
double coord[m_coord][3];
double coord_pre[m_coord][3];
for (int i=0; i<m_coord; i++){
    coord[i][0] = coord_in[i]; coord[i][1] = coord_in[i+m_coord]; coord[i][2] = coord_in[i+m_coord*2]; 
    coord_pre[i][0] = coord[i][0]; coord_pre[i][1] = coord[i][1]; coord_pre[i][2] = coord[i][2]; 
}
//----------------------------------------------------------------------------------------
int id_on_coord[m_id_on_coord];
for (int i=0; i<m_id_on_coord; i++){
    id_on_coord[i] = floor(id_on_coord_in[i]+0.5)-1; 
}
//----------------------------------------------------------------------------------------
int edg[m_A_in][2];
for (int i=0; i<m_A_in; i++){
    edg[i][0] = floor(A_in[i]+0.5)-1; edg[i][1] = floor(A_in[i+m_A_in]+0.5)-1; 
}
int iEdgExo;
for (int i=0; i<m_A_in-1; i++){
    if ((edg[i][0]==edg[m_A_in-1][0]) && (edg[i][1]==edg[m_A_in-1][1])){
        iEdgExo=i;
        break;
    }
}
//----------------------------------------------------------------------------------------
double sMax,dt;
sMax=pm_in[0]; dt=pm_in[1]; 
int nt,i_shift;
nt=floor(pm_in[2]+0.5); i_shift=floor(pm_in[3]+0.5); 
double dr;
dr=pm_in[4]; 
int Non;
Non=pm_in[5]; 
double rl_min,rl_max;
rl_min=pm_in[6]; rl_max=pm_in[7]; 
int local;
local=pm_in[8]; 
int m_A;
if (local==1){
    m_A=m_A_in;}
else {
    m_A=m_A_in-1;
}
double D;
D=pm_in[9]; 
double mu;
mu=pm_in[10]; 
double rd_min,rs_max;
rd_min=pm_in[11],rs_max=pm_in[12];
int rmSeed;
rmSeed=floor(pm_in[13]+0.5);
//----------------------------------------------------------------------------------------  
int in[m_fn];
for (int i=0; i<m_fn; i++){
    in[i] = floor(in_in[i]+0.5)-1; 
}
//----------------------------------------------------------------------------------------
//========================================================================================================================== 
//========================================================================================================================== 
double d[m_A][3];
double r[m_A];
double u[m_A][3];
double f_edg[m_A][3];
double f[m_coord][3];
int nS,nL;
int nSeff,nLeff;
int loc_relaxed;
double rm;
int idTem;
double f_mag, fmagTem;
double rMin=100, rMax=0;
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
for (int it=0; it<nt; it++){
//----------------------------------------------------------------------------------------
rMin=100, rMax=0; //double xyzMax[6];
int mMax;
for (int m=0; m<m_A; m++){
   d[m][0]=coord[edg[m][1]][0]-coord[edg[m][0]][0];
   d[m][1]=coord[edg[m][1]][1]-coord[edg[m][0]][1];
   d[m][2]=coord[edg[m][1]][2]-coord[edg[m][0]][2];
   r[m]=sqrt(d[m][0]*d[m][0]+d[m][1]*d[m][1]+d[m][2]*d[m][2]);
   u[m][0]=d[m][0]/r[m];
   u[m][1]=d[m][1]/r[m];
   u[m][2]=d[m][2]/r[m];
   if (r[m]<rMin){
   rMin=r[m];}
   else if (r[m]>rMax){
   rMax=r[m];
//    xyzMax[0]=coord[edg[m][0]][0]; xyzMax[1]=coord[edg[m][0]][1]; xyzMax[2]=coord[edg[m][0]][2];
//    xyzMax[3]=coord[edg[m][1]][0]; xyzMax[4]=coord[edg[m][1]][1]; xyzMax[5]=coord[edg[m][1]][2];
   mMax=m;
   }
}
// mexPrintf("s1 %d ", local);
//----------------------------------------------------------------------------------------
int ir[m_A];
bool restored=false;
for (int m=0; m<m_A; m++){
  ir[m] = floor(r[m]/dr+0.5)-1-i_shift;
  if ((ir[m] > m_fn-2) || (ir[m]<0)){
      restored=true;
  }
}
if (restored==false){
for (int m=0; m<m_A; m++){
   if ((in[ir[m]]-2<2) || (in[ir[m]]+1>m_rg-3)){
       restored=true;
       break;
   }
}
}
if (restored==true){
    mexPrintf("restored!! before  %f %f\n",rMin,rMax);
    for (int i_id_on=0; i_id_on<m_id_on_coord; i_id_on++){
        int ic=id_on_coord[i_id_on];
        coord[ic][0]=coord_pre[ic][0];
        coord[ic][1]=coord_pre[ic][1];
        coord[ic][2]=coord_pre[ic][2];
    }
    rMin=100, rMax=0;
    for (int m=0; m<m_A; m++){
        d[m][0]=coord[edg[m][1]][0]-coord[edg[m][0]][0];
        d[m][1]=coord[edg[m][1]][1]-coord[edg[m][0]][1];
        d[m][2]=coord[edg[m][1]][2]-coord[edg[m][0]][2];
        r[m]=sqrt(d[m][0]*d[m][0]+d[m][1]*d[m][1]+d[m][2]*d[m][2]);
        u[m][0]=d[m][0]/r[m];
        u[m][1]=d[m][1]/r[m];
        u[m][2]=d[m][2]/r[m];
        if (r[m]<rMin){
            rMin=r[m];}
        else if (r[m]>rMax){
            rMax=r[m];
//    xyzMax[0]=coord[edg[m][0]][0]; xyzMax[1]=coord[edg[m][0]][1]; xyzMax[2]=coord[edg[m][0]][2];
//    xyzMax[3]=coord[edg[m][1]][0]; xyzMax[4]=coord[edg[m][1]][1]; xyzMax[5]=coord[edg[m][1]][2];
            mMax=m;
        }
    }
    mexPrintf("restored!! after  %f %f\n",rMin,rMax);
    for (int m=0; m<m_A; m++){
        ir[m] = floor(r[m]/dr+0.5)-1-i_shift;
    }
}
//----------------------------------------------------------------------------------------    
    outM4[0]=0.;
//     if ((irMin<0)||(irMax>12000)){
//         outM4[0]=1.;
//         mexPrintf("wrong!!! %d %d %f=====***\n",irMin,irMax,dt*1e10);
//         for (int i=0; i<m_coord; i++){
//             mexPrintf("%f %f %f\n",coord[i][0]-coord_pre[i][0],coord[i][1]-coord_pre[i][1],coord[i][2]-coord_pre[i][2]);
//             coord[i][0] = coord_pre[i][0]; coord[i][1] = coord_pre[i][1]; coord[i][2] = coord_pre[i][2];
//         }
//     break;} 
//     else {
//         for (int i=0; i<m_coord; i++){
//             coord_pre[i][0] = coord[i][0]; coord_pre[i][1] = coord[i][1]; coord_pre[i][2] = coord[i][2];
//         }
//     }
//----------------------------------------------------------------------------------------
for (int m=0; m<m_A; m++){
   f_edg[m][0]=fn[ir[m]]*u[m][0];
   f_edg[m][1]=fn[ir[m]]*u[m][1];
   f_edg[m][2]=fn[ir[m]]*u[m][2];
}
//----------------------------------------------------------------------------------------
nS=0; nL=0;
for (int m=0; m<m_A; m++){
    if (r[m]<rl_min) {
    nS++;}
    else if (r[m]>rl_max){
    nL++;}
}
nSeff=nS,nLeff=nL;
int idS[nS], idL[nL];
int idStem=0, idLtem=0;
for (int m=0; m<m_A; m++){
    if (r[m]<rl_min) {
    idS[idStem]=m;  idStem++;}
    else if (r[m]>rl_max){
    idL[idLtem]=m;  idLtem++;}
}
double signS[nS], signL[nL];
for (int iS=0; iS<nS; iS++){
    signS[iS]=1;
    if ((idS[iS]==iEdgExo) && (local==3)){
        if (r[iEdgExo]<rs_max){
            signS[iS]=0;}
        else {
            signS[iS]=-1;}
            nSeff--;
    }
}
for (int iL=0; iL<nL; iL++){
    signL[iL]=1;
    if ((idL[iL]==iEdgExo) && (local==2)){
        if (r[iEdgExo]>rd_min){
            signL[iL]=0;}
        else{
            signL[iL]=-1;}
            nLeff--;
    }
}
// mexPrintf(" s2");
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
loc_relaxed=0;
    if ((local==1) && (nS == 0) && (nL == 0)){
       loc_relaxed = 1;}
    else if ((local==2) && (nS == 0) && (nLeff == 0)){ 
        if (r[iEdgExo]>rd_min){
        loc_relaxed = 1;} 
        else if ((r[iEdgExo]>rl_min) && (r[iEdgExo]<rl_max)){
        loc_relaxed = 2;}
    }
    else if ((local==3) && (nSeff == 0) && (nL == 0)){
        if (r[iEdgExo]<rs_max){
        loc_relaxed = 1;} 
        else if ((r[iEdgExo]>rl_min) && (r[iEdgExo]<rl_max)){
        loc_relaxed = 2;}
    }

    if (loc_relaxed==0) {
        double frandS[nS][3];
    if (nS > 0){
        for (int iS=0; iS<nS; iS++){
            rm=GetUniform();
            idTem=floor((double)(m_RLevy-1)*rm);
            frandS[iS][0] = RLevy[idTem]*u[idS[iS]][0]*D; frandS[iS][1] = RLevy[idTem]*u[idS[iS]][1]*D; frandS[iS][2] = RLevy[idTem]*u[idS[iS]][2]*D;
        }
    }
        
        double frandL[nL][3];
    if (nL > 0){
        for (int iL=0; iL<nL; iL++){
            rm=GetUniform();
            idTem=floor((double)(m_RLevy-1)*rm);
            frandL[iL][0] = -RLevy[idTem]*u[idL[iL]][0]*D; frandL[iL][1] = -RLevy[idTem]*u[idL[iL]][1]*D; frandL[iL][2] = -RLevy[idTem]*u[idL[iL]][2]*D;
        }
    }
//      mexPrintf(" s3");   
    //----------------------------------------------------------------------------------------
    for (int ic=0; ic<m_coord; ic++){
        f[ic][0]=0; f[ic][1]=0; f[ic][2]=0;
    }
    for (int ic=0; ic<m_coord; ic++){
        for (int i_id_on=0; i_id_on<m_id_on_coord; i_id_on++){
          if (id_on_coord[i_id_on]==ic){
          int icOn=ic;
          //---------------------------------------------------------------
          for (int m=0; m<m_A; m++){
              if (edg[m][0]==icOn) {
                  f[icOn][0]-=f_edg[m][0]; f[icOn][1]-=f_edg[m][1]; f[icOn][2]-=f_edg[m][2];
                  for (int iS=0; iS<nS; iS++){
                      if (idS[iS]==m){
                          f[icOn][0]-=frandS[iS][0]*signS[iS]; f[icOn][1]-=frandS[iS][1]*signS[iS]; f[icOn][2]-=frandS[iS][2]*signS[iS];}
                  }
                  for (int iL=0; iL<nL; iL++){
                      if (idL[iL]==m){
                          f[icOn][0]-=frandL[iL][0]*signL[iL]; f[icOn][1]-=frandL[iL][1]*signL[iL]; f[icOn][2]-=frandL[iL][2]*signL[iL];}
                  }
              }
              else if (edg[m][1]==icOn){
                  f[icOn][0]+=f_edg[m][0]; f[icOn][1]+=f_edg[m][1]; f[icOn][2]+=f_edg[m][2];
                  for (int iS=0; iS<nS; iS++){
                      if (idS[iS]==m){
                          f[icOn][0]+=frandS[iS][0]*signS[iS]; f[icOn][1]+=frandS[iS][1]*signS[iS]; f[icOn][2]+=frandS[iS][2]*signS[iS];}
                  }
                  for (int iL=0; iL<nL; iL++){
                      if (idL[iL]==m){
                          f[icOn][0]+=frandL[iL][0]*signL[iL]; f[icOn][1]+=frandL[iL][1]*signL[iL]; f[icOn][2]+=frandL[iL][2]*signL[iL];}
                  }
              }
          }
          //---------------------------------------------------------------
          } 
        }
    }
    }
if (loc_relaxed>0){
    break;
}
else {
//----------------------------------------------------------------------------------------
dt=varDt(m_A, r, ir, d, edg, m_coord, f, dr, mu, local, m_fn, in, m_rg, rg, iEdgExo, restored);
// mexPrintf(" %f\n",dt); 
// double f_mag=sqrt(f[0][0]*f[0][0]+f[0][1]*f[0][1]+f[0][2]*f[0][2]);
// double f_tot=f_mag;
// for (int ic=1; ic<m_coord; ic++){
//     double fmagTem=sqrt(f[ic][0]*f[ic][0]+f[ic][1]*f[ic][1]+f[ic][2]*f[ic][2]);
//     f_tot+=fmagTem;
//     if (fmagTem>f_mag){
//     f_mag=fmagTem;}
// }
for (int i_id_on=0; i_id_on<m_id_on_coord; i_id_on++){
   int ic=id_on_coord[i_id_on]; 
//    coord_pre[ic][0] = coord[ic][0];
//    coord_pre[ic][1] = coord[ic][1]; 
//    coord_pre[ic][2] = coord[ic][2]; 
   coord[ic][0]=coord[ic][0]+f[ic][0]*mu*dt;
   coord[ic][1]=coord[ic][1]+f[ic][1]*mu*dt;
   coord[ic][2]=coord[ic][2]+f[ic][2]*mu*dt;
}
}
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
} //it
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
for (int i=0; i<m_coord; i++){
    outM[i] =  coord[i][0];          outM[i+m_coord] =  coord[i][1];        outM[i+m_coord*2] =  coord[i][2];
    outM3[i] =  f[i][0];             outM3[i+m_coord] =  f[i][1];           outM3[i+m_coord*2] =  f[i][2];
}
outM2[0]=(double)loc_relaxed;
// mexPrintf("======%f %f %d %f;...\n", rMin, rMax, loc_relaxed,r[iEdgExo]);
//----------------------------------------------------------------------------------------
}
//==========================================================================================================================
//==========================================================================================================================
/* The gateway function */
//==========================================================================================================================
//==========================================================================================================================
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
                                                /* get the value of input     get dimensions of the input matrix */
    double *coord;         size_t m_coord;           coord = mxGetPr(prhs[0]);         m_coord = mxGetM(prhs[0]); 
    double *id_on_coord;   size_t m_id_on_coord;     id_on_coord = mxGetPr(prhs[1]);   m_id_on_coord = mxGetM(prhs[1]);
    double *A;             size_t m_A;               A = mxGetPr(prhs[2]);             m_A = mxGetM(prhs[2]);
    double *pm;            size_t m_pm;              pm = mxGetPr(prhs[3]);            m_pm=mxGetM(prhs[3]);
    double *fn;            size_t m_fn;              fn = mxGetPr(prhs[4]);            m_fn=mxGetM(prhs[4]);
    double *RLevy;         size_t m_RLevy;           RLevy = mxGetPr(prhs[5]);         m_RLevy=mxGetM(prhs[5]);
    double *in;                                      in = mxGetPr(prhs[6]);    
    double *rg;            size_t m_rg;              rg = mxGetPr(prhs[7]);            m_rg=mxGetM(prhs[7]);
    
                            /* create the output matrix */                                      /* get a pointer to the real data in the output matrix */
    double *outMatrix;       plhs[0] = mxCreateDoubleMatrix((mwSize)m_coord,3,mxREAL);          outMatrix = mxGetPr(plhs[0]);
    double *outMatrix2;      plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);                        outMatrix2 = mxGetPr(plhs[1]);
    double *outMatrix3;      plhs[2] = mxCreateDoubleMatrix((mwSize)m_coord,3,mxREAL);          outMatrix3 = mxGetPr(plhs[2]);
    double *outMatrix4;      plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);                        outMatrix4 = mxGetPr(plhs[3]);

    /* call the computational routine */
    arrayComp(coord,id_on_coord,A,pm,fn,RLevy,in,rg,
              outMatrix,outMatrix2,outMatrix3,outMatrix4,
              (mwSize)m_coord,(mwSize)m_id_on_coord,(mwSize)m_A,(mwSize)m_fn,(mwSize)m_RLevy, (mwSize)m_rg);

//     for (int i=0; i<m_id_on_coord; i++){
//         mexPrintf("%f ;...\n", id_on_coord[i]);
//     }
}