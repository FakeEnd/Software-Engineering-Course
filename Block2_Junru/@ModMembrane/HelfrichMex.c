#include "mex.h"
#include "matrix.h"
#include "math.h"
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
for (int i=0; i<m_ver; i++){
    ver[i][0] = ver_in[i]; ver[i][1] = ver_in[i+m_ver]; ver[i][2] = ver_in[i+m_ver*2]; 
    //mexPrintf("%f %f %f;...\n", ver[i][0], ver[i][1], ver[i][2]);
}
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
//========================================================================================================================== Helfrich
double A[m_ver]; 
double K[m_ver][3]; double kH[m_ver];
double H[m_ver]; 
for (int m = 0; m < m_ver; m++) {
    kH[m] = 0.; A[m] = 0.;
    for (int k = 0; k < 3; k++) {
        K[m][k] = 0;
    }
    H[m] = 0.;
}
//-------------------------------------------------------------------------
for (int iJ=0; iJ<m_J; iJ++){
    int i = J[iJ];
//------------------------------------------------------------------------- 
    double c[n_node[i]][3]; double b[n_node[i]][3]; double c_d_c[n_node[i]]; double b_d_b[n_node[i]]; double b_d_c[n_node[i]];
    for (int j = 0; j < n_node[i]; j++){
        for (int k = 0; k < 3; k++){
            c[j][k] = ver[i][k]-ver[j_T[i][j]][k];
            b[j][k] = ver[j_T_alt[i][j]][k]-ver[j_T[i][j]][k];
        } 
    }
    for (int j = 0; j < n_node[i]; j++){
        c_d_c[j] = c[j][0]*c[j][0] + c[j][1]*c[j][1] + c[j][2]*c[j][2];
        b_d_b[j] = b[j][0]*b[j][0] + b[j][1]*b[j][1] + b[j][2]*b[j][2];
        b_d_c[j] = b[j][0]*c[j][0] + b[j][1]*c[j][1] + b[j][2]*c[j][2];
    }
    //-------------------------------------------------------------------------
    double b1_d_c2[n_node[i]]; double c2_d_c2[n_node[i]]; double c1_d_c2[n_node[i]];
    for (int j = 0; j < n_node[i]-1; j++){
        b1_d_c2[j] = b[j][0]*c[j+1][0] + b[j][1]*c[j+1][1] + b[j][2]*c[j+1][2];
        c2_d_c2[j] = c[j+1][0]*c[j+1][0] + c[j+1][1]*c[j+1][1] + c[j+1][2]*c[j+1][2];
        c1_d_c2[j] = c[j][0]*c[j+1][0] + c[j][1]*c[j+1][1] + c[j][2]*c[j+1][2];
    }
    b1_d_c2[n_node[i]-1] = b[n_node[i]-1][0]*c[0][0] + b[n_node[i]-1][1]*c[0][1] + b[n_node[i]-1][2]*c[0][2];
    c1_d_c2[n_node[i]-1] = c[n_node[i]-1][0]*c[0][0] + c[n_node[i]-1][1]*c[0][1] + c[n_node[i]-1][2]*c[0][2];
    c2_d_c2[n_node[i]-1] = c[0][0]*c[0][0] + c[0][1]*c[0][1] + c[0][2]*c[0][2];
    //-------------------------------------------------------------------------
    double cosine_abg[n_node[i]][3];
    for (int j = 0; j < n_node[i]; j++){
        cosine_abg[j][0] = b_d_c[j]/(sqrt(b_d_b[j])*sqrt(c_d_c[j]));
        cosine_abg[j][1] = -b1_d_c2[j]/(sqrt(b_d_b[j])*sqrt(c2_d_c2[j]));
        cosine_abg[j][2] = c1_d_c2[j]/(sqrt(c_d_c[j])*sqrt(c2_d_c2[j]));
    }
    double id_tem_all[n_node[i]][3];
    for (int j = 0; j < n_node[i]; j++){
        for (int k = 0; k < 3; k++){
            id_tem_all[j][k] = 0.5+0.5*tanh(100.*(-cosine_abg[j][k]));
        }
    }
    //-------------------------------------------------------------------------
    double A_T_org[n_node[i]]; double A_T[n_node[i]];
    for (int j = 0; j < n_node[i]; j++){
        A_T_org[j] = 0.5*sqrt(c_d_c[j]*b_d_b[j]-b_d_c[j]*b_d_c[j]);
    }
    for (int j = 0; j < n_node[i]; j++){
        A_T[j] = 0.5*A_T_org[j]*id_tem_all[j][2];
        A_T[j] = A_T[j]+0.25*A_T_org[j]*id_tem_all[j][0];
        A_T[j] = A_T[j]+0.25*A_T_org[j]*id_tem_all[j][1];
    }
//     for (int j = 0; j < n_node[i]; j++){
//     mexPrintf("%d %f %f %f;...\n", it, A_T[j], A_T_org[j], A_T_org[j]);
//     }
    //-------------------------------------------------------------------------
    double con_min[n_node[i]]; double id_tem[n_node[i]];
    for (int j = 0; j < n_node[i]; j++){
        con_min[j] = 1;
        for (int k = 0; k < 3; k++){
            if (con_min[j] > cosine_abg[j][k]) {
                con_min[j] = cosine_abg[j][k];
            }
        }
        id_tem[j] = 0.5+0.5*tanh(100.*(con_min[j]));
    }
    //-------------------------------------------------------------------------
    double Dela[n_node[i]]; double Delb[n_node[i]];
    for (int j = 0; j < n_node[i]-1; j++){
         Dela[j] = sqrt(c_d_c[j]*b_d_b[j] - b_d_c[j]*b_d_c[j]);
         Delb[j] = sqrt(c_d_c[j+1]*b_d_b[j]-b1_d_c2[j]*b1_d_c2[j]);
    }
    Dela[n_node[i]-1] = sqrt(c_d_c[n_node[i]-1]*b_d_b[n_node[i]-1] - b_d_c[n_node[i]-1]*b_d_c[n_node[i]-1]);
    Delb[n_node[i]-1] = sqrt(c_d_c[0]*b_d_b[n_node[i]-1]-b1_d_c2[n_node[i]-1]*b1_d_c2[n_node[i]-1]);
    //-------------------------------------------------------------------------
    double cota[n_node[i]]; double cotb[n_node[i]];
    for (int j = 0; j < n_node[i]; j++){
        cota[j] = b_d_c[j]/Dela[j];
        cotb[j] = -b1_d_c2[j]/Delb[j];
    }
    A[i] = 0;
    for (int j = 0; j < n_node[i]-1; j++){
        A[i] += 0.125*((cota[j]+cotb[j+1])*c_d_c[j+1])*id_tem[j]+A_T[j];
    }
    A[i] += (0.125*((cota[n_node[i]-1]+cotb[0])*c_d_c[0])*id_tem[n_node[i]-1]+A_T[n_node[i]-1]);
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j < n_node[i]-1; j++){
        K[i][k] += 0.5/A[i]*((cota[j]+cotb[j+1])*c[j+1][k]); 
        }
        K[i][k] += 0.5/A[i]*((cota[n_node[i]-1]+cotb[0])*c[0][k]); 
    }    
    kH[i] = 0.5*sqrt(K[i][0]*K[i][0]+K[i][1]*K[i][1]+K[i][2]*K[i][2]);
//-------------------------------------------------------------------------
    H[i] = 0.5*pm[3]*(2*kH[i]*2*kH[i])*A[i];
} // iJ
//==========================================================================================================================
//mexPrintf("total fext %f ;...\n",  f_ext_tot);
for (int i=0; i<m_ver; i++){
    outM[i] = H[i]; 
    //mexPrintf("%f ;...\n",  outM4[i]);
}
//-------------------------------------------------------------------------
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
    double *outMatrix;       plhs[0] = mxCreateDoubleMatrix((mwSize)m_ver,1,mxREAL);     outMatrix = mxGetPr(plhs[0]);

    /* call the computational routine */
    arrayComp(ver,pm,edg,face,j_T,J,n_node,T_s,T_e,outMatrix,
              (mwSize)m_ver,(mwSize)m_pm,(mwSize)m_edg,(mwSize)m_face,(mwSize)n_jT,(mwSize)m_J);

//     for (int i=0; i<m_J; i++){
//         mexPrintf("%f ;...\n", J[i]);
//         //mexPrintf("%f %f %f;...\n", ver[i], ver[i+m_ver], ver[i+m_ver*2]);
//     }
}
