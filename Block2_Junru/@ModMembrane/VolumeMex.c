#include "mex.h"
#include "matrix.h"
#include "math.h"
//==========================================================================================================================
//==========================================================================================================================
double getCrossX (double* a, double* b){
    return a[1]*b[2]-a[2]*b[1];
}
double getCrossY (double* a, double* b){
    return -(a[0]*b[2]-a[2]*b[0]);
}
double getCrossZ (double* a, double* b){
    return a[0]*b[1]-a[1]*b[0];
}
//----------------------------------------------------------------------------------------
double getMax (double* x, mwSize nx){
    double xmax;
    xmax = x[0];
    for (int i = 1; i < nx; i++) {
        if ( xmax < x[i] ) {
            xmax = x[i];
        }    
    }
    return xmax;
}
//==========================================================================================================================
double getVtetrahedron (double* x, double* y, double* z, double* O){

double V;
double A[3]; double B[3]; double C[3];
double X[3]; double Y[3]; double Z[3];
double dirV;
double CrossRes[3];
double Edge1[3]; double Edge2[3];
 
    A[0]=x[0]; A[1]=y[0]; A[2]=z[0];
    B[0]=x[1]; B[1]=y[1]; B[2]=z[1];
    C[0]=x[2]; C[1]=y[2]; C[2]=z[2];

    X[0]=x[0]-O[0]; X[1]=x[1]-O[0]; X[2]=x[2]-O[0];
    Y[0]=y[0]-O[1]; Y[1]=y[1]-O[1]; Y[2]=y[2]-O[1];
    Z[0]=z[0]-O[2]; Z[1]=z[1]-O[2]; Z[2]=z[2]-O[2];
    
    V=fabs(-X[2]*Y[1]*Z[0]+X[1]*Y[2]*Z[0]+X[2]*Y[0]*Z[1]-X[0]*Y[2]*Z[1]-X[1]*Y[0]*Z[2]+X[0]*Y[1]*Z[2])/6;
    
    for (int iTem=0; iTem<3; iTem++) {Edge1[iTem]=B[iTem]-A[iTem]; Edge2[iTem]=B[iTem]-C[iTem];}
    
    CrossRes[0] = getCrossX(Edge1,Edge2); CrossRes[1] = getCrossY(Edge1,Edge2); CrossRes[2] = getCrossZ(Edge1,Edge2);
    
    dirV=0; for (int iTem=0; iTem<3; iTem++) { dirV=dirV+CrossRes[iTem]*(A[iTem]-O[iTem]);}
    
    if (dirV>0) {dirV=-1;}       
    else        {dirV=1;}    
    
    V=dirV*V;
    return V;
}
//==========================================================================================================================
//==========================================================================================================================
void arrayComp (double* ver_in, double* pm_in, double* edg, double* face_in, double* j_T_in, double* J_in, double* n_node_in, double* T_s, double* T_e,
        double* outM,
        mwSize m_ver, mwSize m_pm, mwSize m_edg, mwSize m_face, mwSize n_jT, mwSize m_J)
{
//----------------------------------------------------------------------------------------
double pm[m_pm];
//----------------------------------------------------------------------------------------
double dt_s = pm_in[1];
//----------------------------------------------------------------------------------------
double ver[m_ver][3];
double O[3]; 
for (int i=0; i<m_ver; i++){
    ver[i][0] = ver_in[i]; ver[i][1] = ver_in[i+m_ver]; ver[i][2] = ver_in[i+m_ver*2]; 
    if (ver[i][0]<O[0]){
        O[0]=ver[i][0];}
    if (ver[i][1]<O[1]){
        O[1]=ver[i][1];}
    if (ver[i][2]<O[2]){
        O[2]=ver[i][2];}
    //mexPrintf("%f %f %f;...\n", ver[i][0], ver[i][1], ver[i][2]);
}
O[0]-=100; O[1]-=100; O[2]-=100;
//----------------------------------------------------------------------------------------
int face[m_face][3];
for (int i=0; i<m_face; i++){
    for (int k = 0; k < 3; k ++) {
    face[i][k] = floor(face_in[i+m_face*k]+0.5) - 1;
    }
    //mexPrintf("%d %d %d;...\n", face[i][0], face[i][1], face[i][2]);
}
//----------------------------------------------------------------------------------------
int J[m_J];
for (int i=0; i<m_J; i++){
    J[i] = floor(J_in[i]+0.5) - 1;
    //mexPrintf("%d;...\n", J[i]);
}
//----------------------------------------------------------------------------------------
int j_T[m_ver][n_jT]; int j_T_alt[m_ver][n_jT]; int n_node[m_ver]; 
for (int i=0; i<m_ver; i++){
    n_node[i] = floor(n_node_in[i]+0.5);
    for (int j=0; j < n_jT; j ++) {
    j_T[i][j] = floor(j_T_in[i+m_ver*j]+0.5)-1; 
    }
}

for (int i=0; i<m_ver; i++){
    for (int j=0; j < n_jT-1; j ++) {
    j_T_alt[i][j] = j_T[i][j+1]; 
    }
    j_T_alt[i][n_node[i]-1] = j_T[i][0]; 
}
//========================================================================================================================== loading parameter
for (int i = 0; i < m_pm; i++) {
    pm[i] = pm_in[i];
}
//========================================================================================================================== Volume
double VperF[m_face];
for (int m=0; m<m_face; m++){ VperF[m] = 0.;}
double x[3]; double y[3]; double z[3];

for (int iJ=0; iJ<m_J; iJ++){
    int iF = J[iJ];
    x[0]=ver[face[iF][0]][0]; x[1]=ver[face[iF][1]][0]; x[2]=ver[face[iF][2]][0];
    y[0]=ver[face[iF][0]][1]; y[1]=ver[face[iF][1]][1]; y[2]=ver[face[iF][2]][1];
    z[0]=ver[face[iF][0]][2]; z[1]=ver[face[iF][1]][2]; z[2]=ver[face[iF][2]][2];
    
    VperF[iF]=getVtetrahedron(x, y, z, O);
}
//========================================================================================================================== Output
for (int i=0; i<m_face; i++){
    outM[i] = VperF[i];
    //mexPrintf("%f ;...\n",  outM4[i]);
}
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
    double *ver;        size_t m_ver;           ver = mxGetPr(prhs[0]);         m_ver = mxGetM(prhs[0]);
    double *pm;         size_t m_pm;            pm = mxGetPr(prhs[1]);          m_pm = mxGetM(prhs[1]);
    double *edg;        size_t m_edg;           edg = mxGetPr(prhs[2]);         m_edg = mxGetM(prhs[2]);
    double *face;       size_t m_face;          face = mxGetPr(prhs[3]);        m_face = mxGetM(prhs[3]);
    double *j_T;        size_t n_jT;            j_T = mxGetPr(prhs[4]);         n_jT = mxGetN(prhs[4]);
    double *J;          size_t m_J;             J = mxGetPr(prhs[5]);           m_J = mxGetM(prhs[5]); ///J is id_on_coord
    double *n_node;                             n_node = mxGetPr(prhs[6]);
    double *T_s;                                T_s = mxGetPr(prhs[7]);
    double *T_e;                                T_e = mxGetPr(prhs[8]);             
    
                            /* create the output matrix */                              /* get a pointer to the real data in the output matrix */
    double *outMatrix;       plhs[0] = mxCreateDoubleMatrix((mwSize)m_face,1,mxREAL);     outMatrix = mxGetPr(plhs[0]);

    /* call the computational routine */
    arrayComp(ver,pm,edg,face,j_T,J,n_node,T_s,T_e,outMatrix,
              (mwSize)m_ver,(mwSize)m_pm,(mwSize)m_edg,(mwSize)m_face,(mwSize)n_jT,(mwSize)m_J);

//     for (int i=0; i<m_J; i++){
//         mexPrintf("%f ;...\n", J[i]);
//         //mexPrintf("%f %f %f;...\n", ver[i], ver[i+m_ver], ver[i+m_ver*2]);
//     }
}
