#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ static const float kSqrt03_02 = 1.22474487139158894f;
__device__ static const float kSqrt01_03 = 0.57735026918962573f;
__device__ static const float kSqrt02_03 = 0.81649658092772603f;
__device__ static const float kSqrt04_03 = 1.15470053837925146f;
__device__ static const float kSqrt01_04 = 0.50000000000000000f;
__device__ static const float kSqrt03_04 = 0.86602540378443860f;
__device__ static const float kSqrt01_05 = 0.44721359549995793f;
__device__ static const float kSqrt03_05 = 0.77459666924148340f;
__device__ static const float kSqrt06_05 = 1.09544511501033215f;
__device__ static const float kSqrt08_05 = 1.26491106406735176f;
__device__ static const float kSqrt09_05 = 1.34164078649987384f;
__device__ static const float kSqrt05_06 = 0.91287092917527690f;
__device__ static const float kSqrt01_06 = 0.40824829046386302f;
__device__ static const float kSqrt03_08 = 0.61237243569579447f;
__device__ static const float kSqrt05_08 = 0.79056941504209488f;
__device__ static const float kSqrt07_08 = 0.93541434669348533f;
__device__ static const float kSqrt09_08 = 1.06066017177982119f;
__device__ static const float kSqrt05_09 = 0.74535599249992990f;
__device__ static const float kSqrt08_09 = 0.94280904158206336f;

__device__ static const float kSqrt01_10 = 0.31622776601683794f;
__device__ static const float kSqrt03_10 = 0.54772255750516607f;
__device__ static const float kSqrt01_12 = 0.28867513459481287f;
__device__ static const float kSqrt04_15 = 0.51639777949432220f;
__device__ static const float kSqrt01_16 = 0.25000000000000000f;
__device__ static const float kSqrt07_16 = 0.66143782776614768f;
__device__ static const float kSqrt15_16 = 0.96824583655185426f;
__device__ static const float kSqrt01_18 = 0.23570226039551584f;
__device__ static const float kSqrt03_25 = 0.34641016151377546f;
__device__ static const float kSqrt14_25 = 0.74833147735478833f;
__device__ static const float kSqrt15_25 = 0.77459666924148340f;
__device__ static const float kSqrt18_25 = 0.84852813742385702f;
__device__ static const float kSqrt01_32 = 0.17677669529663689f;
__device__ static const float kSqrt03_32 = 0.30618621784789724f;
__device__ static const float kSqrt15_32 = 0.68465319688145765f;
__device__ static const float kSqrt21_32 = 0.81009258730098255f;
__device__ static const float kSqrt01_50 = 0.14142135623730950f;
__device__ static const float kSqrt03_50 = 0.24494897427831780f;
__device__ static const float kSqrt21_50 = 0.64807406984078597f;
__device__ static const float kSqrt1_60 = 0.12909944487358055f;


__device__ void Construct_SH_Rotation_Matrix_Backward(
    //const float * mat,
    const float * gv, const float * v, // 16
    const float sh1[3][3],
    const float sh2[5][5],
    const float sh3[7][7],
    float * g_mat
){
    ////// suppose sh1, sh2, sh3 is computed
    ////// backward, clear data
    float g_sh1[3][3];
    float g_sh2[5][5];
    //float g_sh3[7][7];

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++)
            g_sh1[i][j] = 0;
    }
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++)
            g_sh2[i][j] = 0;
    }
//     for(int i = 0; i < 7; i++){
//         for(int j = 0; j < 7; j++)
//             g_sh3[i][j] = 0;
//     }
    ////// backward
    float tmp;
    // level 3
    tmp = gv[9] * v[9] * kSqrt01_04; // (0,0)
    g_sh1[2][2] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[0][0];
    tmp = gv[9] * v[10] * kSqrt03_02; // (0,1)
    g_sh1[2][1] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[0][1];
    tmp = gv[9] * v[11] * kSqrt15_16; // (0,2)
    g_sh1[2][1] += tmp * sh2[0][1]; g_sh2[0][1] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[4][1]; g_sh2[4][1] += tmp * sh1[0][1];
    tmp = gv[9] * v[12] * kSqrt05_06; // (0,3)
    g_sh1[2][1] += tmp * sh2[0][2]; g_sh2[0][2] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[4][2]; g_sh2[4][2] += tmp * sh1[0][1];
    tmp = gv[9] * v[13] * kSqrt15_16; // (0,4)
    g_sh1[2][1] += tmp * sh2[0][3]; g_sh2[0][3] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[4][3]; g_sh2[4][3] += tmp * sh1[0][1];
    tmp = gv[9] * v[14] * kSqrt03_02; // (0,5)
    g_sh1[2][1] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[0][1];
    tmp = gv[9] * v[15] * kSqrt01_04; // (0,6)
    g_sh1[2][2] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh2[0][0]; g_sh2[0][0] -= tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh2[4][0]; g_sh2[4][0] -= tmp * sh1[0][0];

    tmp = gv[10] * v[9] * kSqrt01_06; // (1,0)
    g_sh1[1][2] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[1][2]; g_sh1[1][0] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[1][0];
    g_sh1[2][2] += tmp * sh2[1][0]; g_sh2[1][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh2[3][0]; g_sh2[3][0] += tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[0][0];
    tmp = gv[10] * v[10]; // (1,1)
    g_sh1[1][1] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[1][0]; g_sh2[1][0] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[3][0]; g_sh2[3][0] += tmp * sh1[0][1];
    tmp = gv[10] * v[11] * kSqrt05_08; // (1,2)
    g_sh1[1][1] += tmp * sh2[0][1]; g_sh2[0][1] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[1][1]; g_sh2[1][1] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[3][1]; g_sh2[3][1] += tmp * sh1[0][1];
    tmp = gv[10] * v[12] * kSqrt05_09; // (1,3)
    g_sh1[1][1] += tmp * sh2[0][2]; g_sh2[0][2] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[1][2]; g_sh2[1][2] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[3][2]; g_sh2[3][2] += tmp * sh1[0][1];
    tmp = gv[10] * v[13] * kSqrt05_08; // (1,4)
    g_sh1[1][1] += tmp * sh2[0][3]; g_sh2[0][3] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[1][3]; g_sh2[1][3] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[3][3]; g_sh2[3][3] += tmp * sh1[0][1];
    tmp = gv[10] * v[14]; // (1,5)
    g_sh1[1][1] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[0][1];
    tmp = gv[10] * v[15] * kSqrt01_06; // (1,6)
    g_sh1[1][2] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[1][2]; g_sh1[1][0] -= tmp * sh2[0][0]; g_sh2[0][0] -= tmp * sh1[1][0];
    g_sh1[2][2] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh2[1][0]; g_sh2[1][0] -= tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh2[3][0]; g_sh2[3][0] -= tmp * sh1[0][0];

    tmp = gv[11] * v[9] * kSqrt04_15; // (2,0)
    g_sh1[1][2] += tmp * sh2[1][0]; g_sh2[1][0] += tmp * sh1[1][2]; g_sh1[1][0] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[1][0];
    tmp = gv[11] * v[9] * kSqrt01_05;
    g_sh1[0][2] += tmp * sh2[2][0]; g_sh2[2][0] += tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[0][0];
    tmp = gv[11] * v[9] * -kSqrt1_60;
    g_sh1[2][2] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[2][0];
    g_sh1[0][2] -= tmp * sh2[4][0]; g_sh2[4][0] -= tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh2[4][4]; g_sh2[4][4] -= tmp * sh1[0][0];
    tmp = gv[11] * v[10] * kSqrt08_05; // (2,1)
    g_sh1[1][1] += tmp * sh2[1][0]; g_sh2[1][0] += tmp * sh1[1][1];
    tmp = gv[11] * v[10] * kSqrt06_05;
    g_sh1[0][1] += tmp * sh2[2][0]; g_sh2[2][0] += tmp * sh1[0][1];
    tmp = gv[11] * v[10] * -kSqrt01_10;
    g_sh1[2][1] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[4][0]; g_sh2[4][0] -= tmp * sh1[0][1];
    tmp = gv[11] * v[11]; // (2,2)
    g_sh1[1][1] += tmp * sh2[1][1]; g_sh2[1][1] += tmp * sh1[1][1];
    tmp = tmp * kSqrt03_04;
    g_sh1[0][1] += tmp * sh2[2][1]; g_sh2[2][1] += tmp * sh1[0][1];
    tmp = gv[11] * v[11] * -kSqrt01_16;
    g_sh1[2][1] += tmp * sh2[0][1]; g_sh2[0][1] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[4][1]; g_sh2[4][1] -= tmp * sh1[0][1];
    tmp = gv[11] * v[12] * kSqrt08_09; // (2,3)
    g_sh1[1][1] += tmp * sh2[1][2]; g_sh2[1][2] += tmp * sh1[1][1];
    tmp = gv[11] * v[12] * kSqrt02_03;
    g_sh1[0][1] += tmp * sh2[2][2]; g_sh2[2][2] += tmp * sh1[0][1];
    tmp = gv[11] * v[12] * -kSqrt01_18;
    g_sh1[2][1] += tmp * sh2[0][2]; g_sh2[0][2] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[4][2]; g_sh2[4][2] -= tmp * sh1[0][1];
    tmp = gv[11] * v[13]; // (2,4)
    g_sh1[1][1] += tmp * sh2[1][3]; g_sh2[1][3] += tmp * sh1[1][1];
    tmp = tmp * kSqrt03_04;
    g_sh1[0][1] += tmp * sh2[2][3]; g_sh2[2][3] += tmp * sh1[0][1];
    tmp = gv[11] * v[13] * -kSqrt01_16;
    g_sh1[2][1] += tmp * sh2[0][3]; g_sh2[0][3] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[4][3]; g_sh2[4][3] -= tmp * sh1[0][1];
    tmp = gv[11] * v[14] * kSqrt08_05; // (2,5)
    g_sh1[1][1] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[1][1];
    tmp = gv[11] * v[14] * kSqrt06_05;
    g_sh1[0][1] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[0][1];
    tmp = gv[11] * v[14] * -kSqrt01_10;
    g_sh1[2][1] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[4][4]; g_sh2[4][4] -= tmp * sh1[0][1];
    tmp = gv[11] * v[15] * kSqrt04_15; // (2,6)
    g_sh1[1][2] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[1][2]; g_sh1[1][0] -= tmp * sh2[1][0]; g_sh2[1][0] -= tmp * sh1[1][0];
    tmp = gv[11] * v[15] * kSqrt01_05;
    g_sh1[0][2] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh2[2][0]; g_sh2[2][0] -= tmp * sh1[0][0];
    tmp = gv[11] * v[15] * -kSqrt1_60;
    g_sh1[2][2] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh2[0][0]; g_sh2[0][0] -= tmp * sh1[2][0];
    g_sh1[0][2] -= tmp * sh2[4][4]; g_sh2[4][4] -= tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[0][0];

    tmp = gv[12] * v[9] * kSqrt03_10; // (3,0)
    g_sh1[1][2] += tmp * sh2[2][0]; g_sh2[2][0] += tmp * sh1[1][2]; g_sh1[1][0] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[1][0];
    tmp = gv[12] * v[9] * -kSqrt01_10;
    g_sh1[2][2] += tmp * sh2[3][0]; g_sh2[3][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh2[1][0]; g_sh2[1][0] += tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[0][0];
    tmp = gv[12] * v[10] * kSqrt09_05; // (3,1)
    g_sh1[1][1] += tmp * sh2[2][0]; g_sh2[2][0] += tmp * sh1[1][1];
    tmp = gv[12] * v[10] * -kSqrt03_05;
    g_sh1[2][1] += tmp * sh2[3][0]; g_sh2[3][0] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[1][0]; g_sh2[1][0] += tmp * sh1[0][1];
    tmp = gv[12] * v[11] * kSqrt09_08; // (3,2)
    g_sh1[1][1] += tmp * sh2[2][1]; g_sh2[2][1] += tmp * sh1[1][1];
    tmp = gv[12] * v[11] * -kSqrt03_08;
    g_sh1[2][1] += tmp * sh2[3][1]; g_sh2[3][1] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[1][1]; g_sh2[1][1] += tmp * sh1[0][1];
    tmp = gv[12] * v[12]; // (3,3)
    g_sh1[1][1] += tmp * sh2[2][2]; g_sh2[2][2] += tmp * sh1[1][1];
    tmp = tmp * -kSqrt01_03;
    g_sh1[2][1] += tmp * sh2[3][2]; g_sh2[3][2] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[1][2]; g_sh2[1][2] += tmp * sh1[0][1];
    tmp = gv[12] * v[13] * kSqrt09_08; // (3,4)
    g_sh1[1][1] += tmp * sh2[2][3]; g_sh2[2][3] += tmp * sh1[1][1];
    tmp = gv[12] * v[13] * -kSqrt03_08;
    g_sh1[2][1] += tmp * sh2[3][3]; g_sh2[3][3] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[1][3]; g_sh2[1][3] += tmp * sh1[0][1];
    tmp = gv[12] * v[14] * kSqrt09_05; // (3,5)
    g_sh1[1][1] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[1][1];
    tmp = gv[12] * v[14] * -kSqrt03_05;
    g_sh1[2][1] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[0][1];
    tmp = gv[12] * v[15] * kSqrt03_10; // (3,6)
    g_sh1[1][2] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[1][2]; g_sh1[1][0] -= tmp * sh2[2][0]; g_sh2[2][0] -= tmp * sh1[1][0];
    tmp = gv[12] * v[15] * -kSqrt01_10;
    g_sh1[2][2] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh2[3][0]; g_sh2[3][0] -= tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh2[1][4]; g_sh2[1][4] += tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh2[1][0]; g_sh2[1][0] -= tmp * sh1[0][0];

    tmp = gv[13] * v[9] * kSqrt04_15; // (4,0)
    g_sh1[1][2] += tmp * sh2[3][0]; g_sh2[3][0] += tmp * sh1[1][2]; g_sh1[1][0] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[1][0];
    tmp = gv[13] * v[9] * kSqrt01_05;
    g_sh1[2][2] += tmp * sh2[2][0]; g_sh2[2][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[2][0];
    tmp = gv[13] * v[9] * -kSqrt1_60;
    g_sh1[2][2] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[0][0];
    tmp = gv[13] * v[10] * kSqrt08_05; // (4,1)
    g_sh1[1][1] += tmp * sh2[3][0]; g_sh2[3][0] += tmp * sh1[1][1];
    tmp = gv[13] * v[10] * kSqrt06_05;
    g_sh1[2][1] += tmp * sh2[2][0]; g_sh2[2][0] += tmp * sh1[2][1];
    tmp = gv[13] * v[10] * -kSqrt01_10;
    g_sh1[2][1] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[0][1];
    tmp = gv[13] * v[11]; // (4,2)
    g_sh1[1][1] += tmp * sh2[3][1]; g_sh2[3][1] += tmp * sh1[1][1];
    tmp = tmp * kSqrt03_04;
    g_sh1[2][1] += tmp * sh2[2][1]; g_sh2[2][1] += tmp * sh1[2][1];
    tmp = gv[13] * v[11] * -kSqrt01_16;
    g_sh1[2][1] += tmp * sh2[4][1]; g_sh2[4][1] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[0][1]; g_sh2[0][1] += tmp * sh1[0][1];
    tmp = gv[13] * v[12] * kSqrt08_09; // (4,3)
    g_sh1[1][1] += tmp * sh2[3][2]; g_sh2[3][2] += tmp * sh1[1][1];
    tmp = gv[13] * v[12] * kSqrt02_03;
    g_sh1[2][1] += tmp * sh2[2][2]; g_sh2[2][2] += tmp * sh1[2][1];
    tmp = gv[13] * v[12] * -kSqrt01_18;
    g_sh1[2][1] += tmp * sh2[4][2]; g_sh2[4][2] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[0][2]; g_sh2[0][2] += tmp * sh1[0][1];
    tmp = gv[13] * v[13]; // (4,4)
    g_sh1[1][1] += tmp * sh2[3][3]; g_sh2[3][3] += tmp * sh1[1][1];
    tmp = tmp * kSqrt03_04;
    g_sh1[2][1] += tmp * sh2[2][3]; g_sh2[2][3] += tmp * sh1[2][1];
    tmp = gv[13] * v[13] * -kSqrt01_16;
    g_sh1[2][1] += tmp * sh2[4][3]; g_sh2[4][3] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[0][3]; g_sh2[0][3] += tmp * sh1[0][1];
    tmp = gv[13] * v[14] * kSqrt08_05; // (4,5)
    g_sh1[1][1] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[1][1];
    tmp = gv[13] * v[14] * kSqrt06_05;
    g_sh1[2][1] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[2][1];
    tmp = gv[13] * v[14] * -kSqrt01_10;
    g_sh1[2][1] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[0][1];
    tmp = gv[13] * v[15] * kSqrt04_15; // (4,6)
    g_sh1[1][2] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[1][2]; g_sh1[1][0] -= tmp * sh2[3][0]; g_sh2[3][0] -= tmp * sh1[1][0];
    tmp = gv[13] * v[15] * kSqrt01_05;
    g_sh1[2][2] += tmp * sh2[2][4]; g_sh2[2][4] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh2[2][0]; g_sh2[2][0] -= tmp * sh1[2][0];
    tmp = gv[13] * v[15] * -kSqrt1_60;
    g_sh1[2][2] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh2[4][0]; g_sh2[4][0] -= tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh2[0][4]; g_sh2[0][4] += tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh2[0][0]; g_sh2[0][0] -= tmp * sh1[0][0];

    tmp = gv[14] * v[9] * kSqrt01_06; // (5,0)
    g_sh1[1][2] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[1][2]; g_sh1[1][0] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[1][0];
    g_sh1[2][2] += tmp * sh2[3][0]; g_sh2[3][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[2][0];
    g_sh1[0][2] -= tmp * sh2[1][0]; g_sh2[1][0] -= tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh2[1][4]; g_sh2[1][4] -= tmp * sh1[0][0];
    tmp = gv[14] * v[10]; // (5,1)
    g_sh1[1][1] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[3][0]; g_sh2[3][0] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[1][0]; g_sh2[1][0] -= tmp * sh1[0][1];
    tmp = gv[14] * v[11] * kSqrt05_08; // (5,2)
    g_sh1[1][1] += tmp * sh2[4][1]; g_sh2[4][1] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[3][1]; g_sh2[3][1] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[1][1]; g_sh2[1][1] -= tmp * sh1[0][1];
    tmp = gv[14] * v[12] * kSqrt05_09; // (5,3)
    g_sh1[1][1] += tmp * sh2[4][2]; g_sh2[4][2] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[3][2]; g_sh2[3][2] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[1][2]; g_sh2[1][2] -= tmp * sh1[0][1];
    tmp = gv[14] * v[13] * kSqrt05_08; // (5,4)
    g_sh1[1][1] += tmp * sh2[4][3]; g_sh2[4][3] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[3][3]; g_sh2[3][3] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[1][3]; g_sh2[1][3] -= tmp * sh1[0][1];
    tmp = gv[14] * v[14]; // (5,5)
    g_sh1[1][1] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[1][4]; g_sh2[1][4] -= tmp * sh1[0][1];
    tmp = gv[14] * v[15] * kSqrt01_06; // (5,6)
    g_sh1[1][2] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[1][2]; g_sh1[1][0] -= tmp * sh2[4][0]; g_sh2[4][0] -= tmp * sh1[1][0];
    g_sh1[2][2] += tmp * sh2[3][4]; g_sh2[3][4] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh2[3][0]; g_sh2[3][0] -= tmp * sh1[2][0];
    g_sh1[0][2] -= tmp * sh2[1][4]; g_sh2[1][4] -= tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh2[1][0]; g_sh2[1][0] += tmp * sh1[0][0];

    tmp = gv[15] * v[9] * kSqrt01_04; // (6,0)
    g_sh1[2][2] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[2][0];
    g_sh1[0][2] -= tmp * sh2[0][0]; g_sh2[0][0] -= tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh2[0][4]; g_sh2[0][4] -= tmp * sh1[0][0];
    tmp = gv[15] * v[10] * kSqrt03_02; // (6,1)
    g_sh1[2][1] += tmp * sh2[4][0]; g_sh2[4][0] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[0][0]; g_sh2[0][0] -= tmp * sh1[0][1];
    tmp = gv[15] * v[11] * kSqrt15_16; // (6,2)
    g_sh1[2][1] += tmp * sh2[4][1]; g_sh2[4][1] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[0][1]; g_sh2[0][1] -= tmp * sh1[0][1];
    tmp = gv[15] * v[12] * kSqrt05_06; // (6,3)
    g_sh1[2][1] += tmp * sh2[4][2]; g_sh2[4][2] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[0][2]; g_sh2[0][2] -= tmp * sh1[0][1];
    tmp = gv[15] * v[13] * kSqrt15_16; // (6,4)
    g_sh1[2][1] += tmp * sh2[4][3]; g_sh2[4][3] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[0][3]; g_sh2[0][3] -= tmp * sh1[0][1];
    tmp = gv[15] * v[14] * kSqrt03_02; // (6,5)
    g_sh1[2][1] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh2[0][4]; g_sh2[0][4] -= tmp * sh1[0][1];
    tmp = gv[15] * v[15] * kSqrt01_04; // (6,6)
    g_sh1[2][2] += tmp * sh2[4][4]; g_sh2[4][4] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh2[4][0]; g_sh2[4][0] -= tmp * sh1[2][0];
    g_sh1[0][2] -= tmp * sh2[0][4]; g_sh2[0][4] -= tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh2[0][0]; g_sh2[0][0] += tmp * sh1[0][0];

    //level 2
    tmp = (gv[4] * v[4] + g_sh2[0][0]) * kSqrt01_04; // (0,0)
    g_sh1[2][2] += tmp * sh1[0][0]; g_sh1[0][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh1[0][2]; g_sh1[0][2] += tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh1[2][0]; g_sh1[2][0] += tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[0][0];
    tmp = (gv[4] * v[5] + g_sh2[0][1]); // (0,1)
    g_sh1[2][1] += tmp * sh1[0][0]; g_sh1[0][0] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh1[2][0]; g_sh1[2][0] += tmp * sh1[0][1];
    tmp = (gv[4] * v[6] + g_sh2[0][2]) * kSqrt03_04; // (0,2)
    g_sh1[2][1] += tmp * sh1[0][1]; g_sh1[0][1] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh1[2][1]; g_sh1[2][1] += tmp * sh1[0][1];
    tmp = (gv[4] * v[7] + g_sh2[0][3]); // (0,3)
    g_sh1[2][1] += tmp * sh1[0][2]; g_sh1[0][2] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[0][1];
    tmp = (gv[4] * v[8] + g_sh2[0][4]) * kSqrt01_04; // (0,4)
    g_sh1[2][2] += tmp * sh1[0][2]; g_sh1[0][2] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh1[0][0]; g_sh1[0][0] -= tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh1[2][0]; g_sh1[2][0] -= tmp * sh1[0][0];

    tmp = (gv[5] * v[4] + g_sh2[1][0]) * kSqrt01_04; // (1,0)
    g_sh1[1][2] += tmp * sh1[0][0]; g_sh1[0][0] += tmp * sh1[1][2]; g_sh1[1][0] += tmp * sh1[0][2]; g_sh1[0][2] += tmp * sh1[1][0];
    g_sh1[0][2] += tmp * sh1[1][0]; g_sh1[1][0] += tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh1[1][2]; g_sh1[1][2] += tmp * sh1[0][0];
    tmp = (gv[5] * v[5] + g_sh2[1][1]); // (1,1)
    g_sh1[1][1] += tmp * sh1[0][0]; g_sh1[0][0] += tmp * sh1[1][1]; g_sh1[0][1] += tmp * sh1[1][0]; g_sh1[1][0] += tmp * sh1[0][1];
    tmp = (gv[5] * v[6] + g_sh2[1][2]) * kSqrt03_04; // (1,2)
    g_sh1[1][1] += tmp * sh1[0][1]; g_sh1[0][1] += tmp * sh1[1][1]; g_sh1[0][1] += tmp * sh1[1][1]; g_sh1[1][1] += tmp * sh1[0][1];
    tmp = (gv[5] * v[7] + g_sh2[1][3]); // (1,3)
    g_sh1[1][1] += tmp * sh1[0][2]; g_sh1[0][2] += tmp * sh1[1][1]; g_sh1[0][1] += tmp * sh1[1][2]; g_sh1[1][2] += tmp * sh1[0][1];
    tmp = (gv[5] * v[8] + g_sh2[1][4]) * kSqrt01_04; // (1,4)
    g_sh1[1][2] += tmp * sh1[0][2]; g_sh1[0][2] += tmp * sh1[1][2]; g_sh1[1][0] -= tmp * sh1[0][0]; g_sh1[0][0] -= tmp * sh1[1][0];
    g_sh1[0][2] += tmp * sh1[1][2]; g_sh1[1][2] += tmp * sh1[0][2]; g_sh1[0][0] -= tmp * sh1[1][0]; g_sh1[1][0] -= tmp * sh1[0][0];

    tmp = (gv[6] * v[4] + g_sh2[2][0]) * kSqrt01_03; // (2,0)
    g_sh1[1][2] += tmp * sh1[1][0]; g_sh1[1][0] += tmp * sh1[1][2]; g_sh1[1][0] += tmp * sh1[1][2]; g_sh1[1][2] += tmp * sh1[1][0];
    tmp = (gv[6] * v[4] + g_sh2[2][0]) * -kSqrt01_12;
    g_sh1[2][2] += tmp * sh1[2][0]; g_sh1[2][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[2][0];
    g_sh1[0][2] += tmp * sh1[0][0]; g_sh1[0][0] += tmp * sh1[0][2]; g_sh1[0][0] += tmp * sh1[0][2]; g_sh1[0][2] += tmp * sh1[0][0];
    tmp = (gv[6] * v[5] + g_sh2[2][1]) * kSqrt04_03; // (2,1)
    g_sh1[1][1] += tmp * sh1[1][0]; g_sh1[1][0] += tmp * sh1[1][1];
    tmp = (gv[6] * v[5] + g_sh2[2][1]) * -kSqrt01_03;
    g_sh1[2][1] += tmp * sh1[2][0]; g_sh1[2][0] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh1[0][0]; g_sh1[0][0] += tmp * sh1[0][1];
    tmp = (gv[6] * v[6] + g_sh2[2][2]);  g_sh1[1][1] += tmp * sh1[1][1] * 2; //(2,2)
    tmp = tmp * -kSqrt01_04; //tmp = gv[6] * v[6] * -kSqrt01_04;
    g_sh1[2][1] += tmp * sh1[2][1] * 2; g_sh1[0][1] += tmp * sh1[0][1] * 2;
    tmp = (gv[6] * v[7] + g_sh2[2][3]) * kSqrt04_03; // (2,3)
    g_sh1[1][1] += tmp * sh1[1][2]; g_sh1[1][2] += tmp * sh1[1][1];
    tmp = (gv[6] * v[7] + g_sh2[2][3]) * -kSqrt01_03;
    g_sh1[2][1] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[2][1]; g_sh1[0][1] += tmp * sh1[0][2]; g_sh1[0][2] += tmp * sh1[0][1];
    tmp = (gv[6] * v[8] + g_sh2[2][4]) * kSqrt01_03; // (2,4)
    g_sh1[1][2] += tmp * sh1[1][2] * 2; g_sh1[1][0] -= tmp * sh1[1][0] * 2;
    tmp = (gv[6] * v[8] + g_sh2[2][4]) * -kSqrt01_12;
    g_sh1[2][2] += tmp * sh1[2][2] * 2; g_sh1[2][0] -= tmp * sh1[2][0] * 2; g_sh1[0][2] += tmp * sh1[0][2] * 2; g_sh1[0][0] -= tmp * sh1[0][0] * 2;

    tmp = (gv[7] * v[4] + g_sh2[3][0]) * kSqrt01_04; // (3,0)
    g_sh1[1][2] += tmp * sh1[2][0]; g_sh1[2][0] += tmp * sh1[1][2]; g_sh1[1][0] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[1][0];
    g_sh1[2][2] += tmp * sh1[1][0]; g_sh1[1][0] += tmp * sh1[2][2]; g_sh1[2][0] += tmp * sh1[1][2]; g_sh1[1][2] += tmp * sh1[2][0];
    tmp = (gv[7] * v[5] + g_sh2[3][1]); // (3,1)
    g_sh1[1][1] += tmp * sh1[2][0]; g_sh1[2][0] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh1[1][0]; g_sh1[1][0] += tmp * sh1[2][1];
    tmp = (gv[7] * v[6] + g_sh2[3][2]) * kSqrt03_04; // (3,2)
    g_sh1[1][1] += tmp * sh1[2][1]; g_sh1[2][1] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh1[1][1]; g_sh1[1][1] += tmp * sh1[2][1];
    tmp = (gv[7] * v[7] + g_sh2[3][3]); // (3,3)
    g_sh1[1][1] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[1][1]; g_sh1[2][1] += tmp * sh1[1][2]; g_sh1[1][2] += tmp * sh1[2][1];
    tmp = (gv[7] * v[8] + g_sh2[3][4]) * kSqrt01_04; // (3,4)
    g_sh1[1][2] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[1][2]; g_sh1[1][0] -= tmp * sh1[2][0]; g_sh1[2][0] -= tmp * sh1[1][0];
    g_sh1[2][2] += tmp * sh1[1][2]; g_sh1[1][2] += tmp * sh1[2][2]; g_sh1[2][0] -= tmp * sh1[1][0]; g_sh1[1][0] -= tmp * sh1[2][0];

    tmp = (gv[8] * v[4] + g_sh2[4][0]) * kSqrt01_04; // (4,0)
    g_sh1[2][2] += tmp * sh1[2][0] * 2; g_sh1[2][0] += tmp * sh1[2][2] * 2;
    g_sh1[0][2] -= tmp * sh1[0][0] * 2; g_sh1[0][0] -= tmp * sh1[0][2] * 2;
    tmp = (gv[8] * v[5] + g_sh2[4][1]); // (4,1)
    g_sh1[2][1] += tmp * sh1[2][0]; g_sh1[2][0] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh1[0][0]; g_sh1[0][0] -= tmp * sh1[0][1];
    tmp = (gv[8] * v[6] + g_sh2[4][2]) * kSqrt03_04; // (4,2)
    g_sh1[2][1] += tmp * sh1[2][1] * 2; g_sh1[0][1] -= tmp * sh1[0][1] * 2;
    tmp = (gv[8] * v[7] + g_sh2[4][3]); // (4,3)
    g_sh1[2][1] += tmp * sh1[2][2]; g_sh1[2][2] += tmp * sh1[2][1]; g_sh1[0][1] -= tmp * sh1[0][2]; g_sh1[0][2] -= tmp * sh1[0][1];
    tmp = (gv[8] * v[8] + g_sh2[4][4]) * kSqrt01_04; // (4,4)
    g_sh1[2][2] += tmp * sh1[2][2] * 2; g_sh1[2][0] -= tmp * sh1[2][0] * 2; g_sh1[0][2] -= tmp * sh1[0][2] * 2; g_sh1[0][0] += tmp * sh1[0][0] * 2;

    // level 1
    g_mat[4] = g_sh1[0][0] + gv[1] * v[1]; g_mat[5] = g_sh1[0][1] + gv[1] * v[2]; g_mat[3] = g_sh1[0][2] + gv[1] * v[3];
    g_mat[7] = g_sh1[1][0] + gv[2] * v[1]; g_mat[8] = g_sh1[1][1] + gv[2] * v[2]; g_mat[6] = g_sh1[1][2] + gv[2] * v[3];
    g_mat[1] = g_sh1[2][0] + gv[3] * v[1]; g_mat[2] = g_sh1[2][1] + gv[3] * v[2]; g_mat[0] = g_sh1[2][2] + gv[3] * v[3];
}


__device__ void Construct_SH_Rotation_Matrix(const float* mat,
	float sh1[3][3],
	float sh2[5][5],
	float sh3[7][7]
) {
    // level 1
	sh1[0][0] = mat[4]; sh1[0][1] = mat[5]; sh1[0][2] = mat[3];
	sh1[1][0] = mat[7]; sh1[1][1] = mat[8]; sh1[1][2] = mat[6];
	sh1[2][0] = mat[1]; sh1[2][1] = mat[2]; sh1[2][2] = mat[0];

    // level 2
    sh2[0][0] = kSqrt01_04 * ((sh1[2][2] * sh1[0][0] + sh1[2][0] * sh1[0][2]) + (sh1[0][2] * sh1[2][0] + sh1[0][0] * sh1[2][2]));
    sh2[0][1] = (sh1[2][1] * sh1[0][0] + sh1[0][1] * sh1[2][0]);
    sh2[0][2] = kSqrt03_04 * (sh1[2][1] * sh1[0][1] + sh1[0][1] * sh1[2][1]);
    sh2[0][3] = (sh1[2][1] * sh1[0][2] + sh1[0][1] * sh1[2][2]);
    sh2[0][4] = kSqrt01_04 * ((sh1[2][2] * sh1[0][2] - sh1[2][0] * sh1[0][0]) + (sh1[0][2] * sh1[2][2] - sh1[0][0] * sh1[2][0]));

    sh2[1][0] = kSqrt01_04 * ((sh1[1][2] * sh1[0][0] + sh1[1][0] * sh1[0][2]) + (sh1[0][2] * sh1[1][0] + sh1[0][0] * sh1[1][2]));
    sh2[1][1] = sh1[1][1] * sh1[0][0] + sh1[0][1] * sh1[1][0];
    sh2[1][2] = kSqrt03_04 * (sh1[1][1] * sh1[0][1] + sh1[0][1] * sh1[1][1]);
    sh2[1][3] = sh1[1][1] * sh1[0][2] + sh1[0][1] * sh1[1][2];
    sh2[1][4] = kSqrt01_04 * ((sh1[1][2] * sh1[0][2] - sh1[1][0] * sh1[0][0]) + (sh1[0][2] * sh1[1][2] - sh1[0][0] * sh1[1][0]));

    sh2[2][0] = kSqrt01_03 * (sh1[1][2] * sh1[1][0] + sh1[1][0] * sh1[1][2]) + -kSqrt01_12 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) + (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]));
    sh2[2][1] = kSqrt04_03 * sh1[1][1] * sh1[1][0] + -kSqrt01_03 * (sh1[2][1] * sh1[2][0] + sh1[0][1] * sh1[0][0]);
    sh2[2][2] = sh1[1][1] * sh1[1][1] + -kSqrt01_04 * (sh1[2][1] * sh1[2][1] + sh1[0][1] * sh1[0][1]);
    sh2[2][3] = kSqrt04_03 * sh1[1][1] * sh1[1][2] + -kSqrt01_03 * (sh1[2][1] * sh1[2][2] + sh1[0][1] * sh1[0][2]);
    sh2[2][4] = kSqrt01_03 * (sh1[1][2] * sh1[1][2] - sh1[1][0] * sh1[1][0]) + -kSqrt01_12 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) + (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]));

    sh2[3][0] = kSqrt01_04 * ((sh1[1][2] * sh1[2][0] + sh1[1][0] * sh1[2][2]) + (sh1[2][2] * sh1[1][0] + sh1[2][0] * sh1[1][2]));
    sh2[3][1] = sh1[1][1] * sh1[2][0] + sh1[2][1] * sh1[1][0];
    sh2[3][2] = kSqrt03_04 * (sh1[1][1] * sh1[2][1] + sh1[2][1] * sh1[1][1]);
    sh2[3][3] = sh1[1][1] * sh1[2][2] + sh1[2][1] * sh1[1][2];
    sh2[3][4] = kSqrt01_04 * ((sh1[1][2] * sh1[2][2] - sh1[1][0] * sh1[2][0]) + (sh1[2][2] * sh1[1][2] - sh1[2][0] * sh1[1][0]));

    sh2[4][0] = kSqrt01_04 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) - (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]));
    sh2[4][1] = (sh1[2][1] * sh1[2][0] - sh1[0][1] * sh1[0][0]);
    sh2[4][2] = kSqrt03_04 * (sh1[2][1] * sh1[2][1] - sh1[0][1] * sh1[0][1]);
    sh2[4][3] = (sh1[2][1] * sh1[2][2] - sh1[0][1] * sh1[0][2]);
    sh2[4][4] = kSqrt01_04 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) - (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]));

    // level 3
    sh3[0][0] = kSqrt01_04 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) + (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]));
    sh3[0][1] = kSqrt03_02 * (sh1[2][1] * sh2[0][0] + sh1[0][1] * sh2[4][0]);
    sh3[0][2] = kSqrt15_16 * (sh1[2][1] * sh2[0][1] + sh1[0][1] * sh2[4][1]);
    sh3[0][3] = kSqrt05_06 * (sh1[2][1] * sh2[0][2] + sh1[0][1] * sh2[4][2]);
    sh3[0][4] = kSqrt15_16 * (sh1[2][1] * sh2[0][3] + sh1[0][1] * sh2[4][3]);
    sh3[0][5] = kSqrt03_02 * (sh1[2][1] * sh2[0][4] + sh1[0][1] * sh2[4][4]);
    sh3[0][6] = kSqrt01_04 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) + (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]));

    sh3[1][0] = kSqrt01_06 * (sh1[1][2] * sh2[0][0] + sh1[1][0] * sh2[0][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][0] + sh1[2][0] * sh2[1][4]) + (sh1[0][2] * sh2[3][0] + sh1[0][0] * sh2[3][4]));
    sh3[1][1] = sh1[1][1] * sh2[0][0] + (sh1[2][1] * sh2[1][0] + sh1[0][1] * sh2[3][0]);
    sh3[1][2] = kSqrt05_08 * sh1[1][1] * sh2[0][1] + kSqrt05_08 * (sh1[2][1] * sh2[1][1] + sh1[0][1] * sh2[3][1]);
    sh3[1][3] = kSqrt05_09 * sh1[1][1] * sh2[0][2] + kSqrt05_09 * (sh1[2][1] * sh2[1][2] + sh1[0][1] * sh2[3][2]);
    sh3[1][4] = kSqrt05_08 * sh1[1][1] * sh2[0][3] + kSqrt05_08 * (sh1[2][1] * sh2[1][3] + sh1[0][1] * sh2[3][3]);
    sh3[1][5] = sh1[1][1] * sh2[0][4] + (sh1[2][1] * sh2[1][4] + sh1[0][1] * sh2[3][4]);
    sh3[1][6] = kSqrt01_06 * (sh1[1][2] * sh2[0][4] - sh1[1][0] * sh2[0][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][4] - sh1[2][0] * sh2[1][0]) + (sh1[0][2] * sh2[3][4] - sh1[0][0] * sh2[3][0]));

    sh3[2][0] = kSqrt04_15 * (sh1[1][2] * sh2[1][0] + sh1[1][0] * sh2[1][4]) + kSqrt01_05 * (sh1[0][2] * sh2[2][0] + sh1[0][0] * sh2[2][4]) + -kSqrt1_60 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) - (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]));
    sh3[2][1] = kSqrt08_05 * sh1[1][1] * sh2[1][0] + kSqrt06_05 * sh1[0][1] * sh2[2][0] + -kSqrt01_10 * (sh1[2][1] * sh2[0][0] - sh1[0][1] * sh2[4][0]);
    sh3[2][2] = sh1[1][1] * sh2[1][1] + kSqrt03_04 * sh1[0][1] * sh2[2][1] + -kSqrt01_16 * (sh1[2][1] * sh2[0][1] - sh1[0][1] * sh2[4][1]);
    sh3[2][3] = kSqrt08_09 * sh1[1][1] * sh2[1][2] + kSqrt02_03 * sh1[0][1] * sh2[2][2] + -kSqrt01_18 * (sh1[2][1] * sh2[0][2] - sh1[0][1] * sh2[4][2]);
    sh3[2][4] = sh1[1][1] * sh2[1][3] + kSqrt03_04 * sh1[0][1] * sh2[2][3] + -kSqrt01_16 * (sh1[2][1] * sh2[0][3] - sh1[0][1] * sh2[4][3]);
    sh3[2][5] = kSqrt08_05 * sh1[1][1] * sh2[1][4] + kSqrt06_05 * sh1[0][1] * sh2[2][4] + -kSqrt01_10 * (sh1[2][1] * sh2[0][4] - sh1[0][1] * sh2[4][4]);
    sh3[2][6] = kSqrt04_15 * (sh1[1][2] * sh2[1][4] - sh1[1][0] * sh2[1][0]) + kSqrt01_05 * (sh1[0][2] * sh2[2][4] - sh1[0][0] * sh2[2][0]) + -kSqrt1_60 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) - (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]));

    sh3[3][0] = kSqrt03_10 * (sh1[1][2] * sh2[2][0] + sh1[1][0] * sh2[2][4]) + -kSqrt01_10 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) + (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]));
    sh3[3][1] = kSqrt09_05 * sh1[1][1] * sh2[2][0] + -kSqrt03_05 * (sh1[2][1] * sh2[3][0] + sh1[0][1] * sh2[1][0]);
    sh3[3][2] = kSqrt09_08 * sh1[1][1] * sh2[2][1] + -kSqrt03_08 * (sh1[2][1] * sh2[3][1] + sh1[0][1] * sh2[1][1]);
    sh3[3][3] = sh1[1][1] * sh2[2][2] + -kSqrt01_03 * (sh1[2][1] * sh2[3][2] + sh1[0][1] * sh2[1][2]);
    sh3[3][4] = kSqrt09_08 * sh1[1][1] * sh2[2][3] + -kSqrt03_08 * (sh1[2][1] * sh2[3][3] + sh1[0][1] * sh2[1][3]);
    sh3[3][5] = kSqrt09_05 * sh1[1][1] * sh2[2][4] + -kSqrt03_05 * (sh1[2][1] * sh2[3][4] + sh1[0][1] * sh2[1][4]);
    sh3[3][6] = kSqrt03_10 * (sh1[1][2] * sh2[2][4] - sh1[1][0] * sh2[2][0]) + -kSqrt01_10 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) + (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]));

    sh3[4][0] = kSqrt04_15 * (sh1[1][2] * sh2[3][0] + sh1[1][0] * sh2[3][4]) + kSqrt01_05 * (sh1[2][2] * sh2[2][0] + sh1[2][0] * sh2[2][4]) + -kSqrt1_60 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) + (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]));
    sh3[4][1] = kSqrt08_05 * sh1[1][1] * sh2[3][0] + kSqrt06_05 * sh1[2][1] * sh2[2][0] + -kSqrt01_10 * (sh1[2][1] * sh2[4][0] + sh1[0][1] * sh2[0][0]);
    sh3[4][2] = sh1[1][1] * sh2[3][1] + kSqrt03_04 * sh1[2][1] * sh2[2][1] + -kSqrt01_16 * (sh1[2][1] * sh2[4][1] + sh1[0][1] * sh2[0][1]);
    sh3[4][3] = kSqrt08_09 * sh1[1][1] * sh2[3][2] + kSqrt02_03 * sh1[2][1] * sh2[2][2] + -kSqrt01_18 * (sh1[2][1] * sh2[4][2] + sh1[0][1] * sh2[0][2]);
    sh3[4][4] = sh1[1][1] * sh2[3][3] + kSqrt03_04 * sh1[2][1] * sh2[2][3] + -kSqrt01_16 * (sh1[2][1] * sh2[4][3] + sh1[0][1] * sh2[0][3]);
    sh3[4][5] = kSqrt08_05 * sh1[1][1] * sh2[3][4] + kSqrt06_05 * sh1[2][1] * sh2[2][4] + -kSqrt01_10 * (sh1[2][1] * sh2[4][4] + sh1[0][1] * sh2[0][4]);
    sh3[4][6] = kSqrt04_15 * (sh1[1][2] * sh2[3][4] - sh1[1][0] * sh2[3][0]) + kSqrt01_05 * (sh1[2][2] * sh2[2][4] - sh1[2][0] * sh2[2][0]) + -kSqrt1_60 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) + (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]));

    sh3[5][0] = kSqrt01_06 * (sh1[1][2] * sh2[4][0] + sh1[1][0] * sh2[4][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) - (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]));
    sh3[5][1] = sh1[1][1] * sh2[4][0] + (sh1[2][1] * sh2[3][0] - sh1[0][1] * sh2[1][0]);
    sh3[5][2] = kSqrt05_08 * sh1[1][1] * sh2[4][1] + kSqrt05_08 * (sh1[2][1] * sh2[3][1] - sh1[0][1] * sh2[1][1]);
    sh3[5][3] = kSqrt05_09 * sh1[1][1] * sh2[4][2] + kSqrt05_09 * (sh1[2][1] * sh2[3][2] - sh1[0][1] * sh2[1][2]);
    sh3[5][4] = kSqrt05_08 * sh1[1][1] * sh2[4][3] + kSqrt05_08 * (sh1[2][1] * sh2[3][3] - sh1[0][1] * sh2[1][3]);
    sh3[5][5] = sh1[1][1] * sh2[4][4] + (sh1[2][1] * sh2[3][4] - sh1[0][1] * sh2[1][4]);
    sh3[5][6] = kSqrt01_06 * (sh1[1][2] * sh2[4][4] - sh1[1][0] * sh2[4][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) - (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]));

    sh3[6][0] = kSqrt01_04 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) - (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]));
    sh3[6][1] = kSqrt03_02 * (sh1[2][1] * sh2[4][0] - sh1[0][1] * sh2[0][0]);
    sh3[6][2] = kSqrt15_16 * (sh1[2][1] * sh2[4][1] - sh1[0][1] * sh2[0][1]);
    sh3[6][3] = kSqrt05_06 * (sh1[2][1] * sh2[4][2] - sh1[0][1] * sh2[0][2]);
    sh3[6][4] = kSqrt15_16 * (sh1[2][1] * sh2[4][3] - sh1[0][1] * sh2[0][3]);
    sh3[6][5] = kSqrt03_02 * (sh1[2][1] * sh2[4][4] - sh1[0][1] * sh2[0][4]);
    sh3[6][6] = kSqrt01_04 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) - (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]));
}

//////////////////

__device__ void InitRot3x3(float sh[3][3]){
    #pragma unroll
    for(int i = 0; i < 3; i++){
        #pragma unroll
        for(int j = 0; j < 3; j++){
            sh[i][j] = 0;
        }
    }
}

__device__ void InitRot5x5(float sh[5][5]){
    #pragma unroll
    for(int i = 0; i < 5; i++){
        #pragma unroll
        for(int j = 0; j < 5; j++){
            sh[i][j] = 0;
        }
    }
}

__device__ void InitRot7x7(float sh[7][7]){
    #pragma unroll
    for(int i = 0; i < 7; i++){
        #pragma unroll
        for(int j = 0; j < 7; j++){
            sh[i][j] = 0;
        }
    }
}

////////////

__device__ void MADRot3x3(float sh[3][3], const float sh_[3][3], float k){
    #pragma unroll
    for(int i = 0; i < 3; i++){
        #pragma unroll
        for(int j =0; j < 3; j++){
            sh[i][j] += sh_[i][j] * k;
        }
    }
}

__device__ void MADRot5x5(float sh[5][5], const float sh_[5][5], float k){
    #pragma unroll
    for(int i = 0; i < 5; i++){
        #pragma unroll
        for(int j =0; j < 5; j++){
            sh[i][j] += sh_[i][j] * k;
        }
    }
}

__device__ void MADRot7x7(float sh[7][7], const float sh_[7][7], float k){
    #pragma unroll
    for(int i = 0; i < 7; i++){
        #pragma unroll
        for(int j =0; j < 7; j++){
            sh[i][j] += sh_[i][j] * k;
        }
    }
}


