#include <vector_types.h>

/* scalar functions used in vector functions */
#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif

//struct __device_builtin__ float2
//{
//	float x, y;
//};
//
//struct __device_builtin__ float3
//{
//	float x, y, z;
//};

//__forceinline__ __device__ int max(int a, int b)
//{
//	return a > b ? a : b;
//}
//
//__forceinline__ __device__ int min(int a, int b)
//{
//	return a < b ? a : b;
//}

//__forceinline__ __device__ long long max(long long a, long long b)
//{
//	return a > b ? a : b;
//}
//
//__forceinline__ __device__ long long min(long long a, long long b)
//{
//	return a < b ? a : b;
//}
//
//__forceinline__ __device__ unsigned int max(unsigned int a, unsigned int b)
//{
//	return a > b ? a : b;
//}
//
//__forceinline__ __device__ unsigned int min(unsigned int a, unsigned int b)
//{
//	return a < b ? a : b;
//}
//
//__forceinline__ __device__ unsigned long long max(unsigned long long a, unsigned long long b)
//{
//	return a > b ? a : b;
//}
//
//__forceinline__ __device__ unsigned long long min(unsigned long long a, unsigned long long b)
//{
//	return a < b ? a : b;
//}


/** lerp */
__forceinline__ __device__ float lerp(const float a, const float b, const float t)
{
	return a + t * (b - a);
}

/** bilerp */
__forceinline__ __device__ float bilerp(const float x00, const float x10, const float x01, const float x11,
	const float u, const float v)
{
	return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

__forceinline__ __device__ float clamp(const float f, const float a, const float b)
{
	return fmaxf(a, fminf(f, b));
}


/* float2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
__forceinline__ __device__ float2 make_float2(const float a, const float b)
{
	float2 s2;
	s2.x = a;
	s2.y = b;
	return s2;
}
__forceinline__ __device__ float2 make_float2(const float s)
{
	return make_float2(s, s);
}
__forceinline__ __device__ float2 make_float2(const int2& a)
{
	return make_float2(float(a.x), float(a.y));
}
__forceinline__ __device__ float2 make_float2(const uint2& a)
{
	return make_float2(float(a.x), float(a.y));
}
/** @} */

/** negate */
__forceinline__ __device__ float2 operator-(const float2& a)
{
	return make_float2(-a.x, -a.y);
}

/** min
* @{
*/
__forceinline__ __device__ float2 fminf(const float2& a, const float2& b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
__forceinline__ __device__ float fminf(const float2& a)
{
	return fminf(a.x, a.y);
}
/** @} */

/** max
* @{
*/
__forceinline__ __device__ float2 fmaxf(const float2& a, const float2& b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
__forceinline__ __device__ float fmaxf(const float2& a)
{
	return fmaxf(a.x, a.y);
}
/** @} */

/** add
* @{
*/
__forceinline__ __device__ float2 operator+(const float2& a, const float2& b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
__forceinline__ __device__ float2 operator+(const float2& a, const float b)
{
	return make_float2(a.x + b, a.y + b);
}
__forceinline__ __device__ float2 operator+(const float a, const float2& b)
{
	return make_float2(a + b.x, a + b.y);
}
__forceinline__ __device__ void operator+=(float2& a, const float2& b)
{
	a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
__forceinline__ __device__ float2 operator-(const float2& a, const float2& b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
__forceinline__ __device__ float2 operator-(const float2& a, const float b)
{
	return make_float2(a.x - b, a.y - b);
}
__forceinline__ __device__ float2 operator-(const float a, const float2& b)
{
	return make_float2(a - b.x, a - b.y);
}
__forceinline__ __device__ void operator-=(float2& a, const float2& b)
{
	a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
__forceinline__ __device__ float2 operator*(const float2& a, const float2& b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
__forceinline__ __device__ float2 operator*(const float2& a, const float s)
{
	return make_float2(a.x * s, a.y * s);
}
__forceinline__ __device__ float2 operator*(const float s, const float2& a)
{
	return make_float2(a.x * s, a.y * s);
}
__forceinline__ __device__ void operator*=(float2& a, const float2& s)
{
	a.x *= s.x; a.y *= s.y;
}
__forceinline__ __device__ void operator*=(float2& a, const float s)
{
	a.x *= s; a.y *= s;
}
/** @} */

/** divide
* @{
*/
__forceinline__ __device__ float2 operator/(const float2& a, const float2& b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}
__forceinline__ __device__ float2 operator/(const float2& a, const float s)
{
	float inv = 1.0f / s;
	return a * inv;
}
__forceinline__ __device__ float2 operator/(const float s, const float2& a)
{
	return make_float2(s / a.x, s / a.y);
}
__forceinline__ __device__ void operator/=(float2& a, const float s)
{
	float inv = 1.0f / s;
	a *= inv;
}
/** @} */

/** lerp */
__forceinline__ __device__ float2 lerp(const float2& a, const float2& b, const float t)
{
	return a + t * (b - a);
}

/** bilerp */
__forceinline__ __device__ float2 bilerp(const float2& x00, const float2& x10, const float2& x01, const float2& x11,
	const float u, const float v)
{
	return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
* @{
*/
__forceinline__ __device__ float2 clamp(const float2& v, const float a, const float b)
{
	return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

__forceinline__ __device__ float2 clamp(const float2& v, const float2& a, const float2& b)
{
	return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** dot product */
__forceinline__ __device__ float dot(const float2& a, const float2& b)
{
	return a.x * b.x + a.y * b.y;
}

/** length */
__forceinline__ __device__ float length(const float2& v)
{
	return sqrtf(dot(v, v));
}

__forceinline__ __device__ float2 grad_length(const float2 & v, float grad) {
	float L = sqrtf(dot(v, v));
	return v * (grad / L);
}

/** normalize */
__forceinline__ __device__ float2 normalize(const float2& v)
{
	float invLen = 1.0f / sqrtf(dot(v, v));
	return v * invLen;
}

__forceinline__ __device__ float2 grad_normalize(const float2 & v, const float2 & grad_output)
{
	float v2 = dot(v, v);
	float v1_5 = sqrtf(v2) * v2; // (x^2+y^2+z^2)^(+3/2)
	float gx = grad_output.x * (v.y * v.y) - grad_output.y * (v.x * v.y);
	float gy = -grad_output.x * (v.x * v.y) + grad_output.y * (v.x * v.x);
	return make_float2(gx, gy) / v1_5;
}

/** floor */
__forceinline__ __device__ float2 floor(const float2& v)
{
	return make_float2(::floorf(v.x), ::floorf(v.y));
}

/** reflect */
__forceinline__ __device__ float2 reflect(const float2& i, const float2& n)
{
	return i - 2.0f * n * dot(n, i);
}


/* float3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
__forceinline__ __device__ float3 make_float3(const float a, const float b, const float c)
{
	float3 s3;
	s3.x = a;
	s3.y = b;
	s3.z = c;
	return s3;
}
__forceinline__ __device__ float3 make_float3(const float s)
{
	return make_float3(s, s, s);
}
__forceinline__ __device__ float3 make_float3(const float2& a)
{
	return make_float3(a.x, a.y, 0.0f);
}
__forceinline__ __device__ float3 make_float3(const int3& a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}
__forceinline__ __device__ float3 make_float3(const uint3& a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}
/** @} */

/** negate */
__forceinline__ __device__ float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

/** min
* @{
*/
__forceinline__ __device__ float3 fminf(const float3& a, const float3& b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__forceinline__ __device__ float fminf(const float3& a)
{
	return fminf(fminf(a.x, a.y), a.z);
}
/** @} */

/** max
* @{
*/
__forceinline__ __device__ float3 fmaxf(const float3& a, const float3& b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__forceinline__ __device__ float fmaxf(const float3& a)
{
	return fmaxf(fmaxf(a.x, a.y), a.z);
}
/** @} */

/** add
* @{
*/
__forceinline__ __device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __device__ float3 operator+(const float3& a, const float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __device__ float3 operator+(const float a, const float3& b)
{
	return make_float3(a + b.x, a + b.y, a + b.z);
}
__forceinline__ __device__ void operator+=(float3& a, const float3& b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
__forceinline__ __device__ float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __device__ float3 operator-(const float3& a, const float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __device__ float3 operator-(const float a, const float3& b)
{
	return make_float3(a - b.x, a - b.y, a - b.z);
}
__forceinline__ __device__ void operator-=(float3& a, const float3& b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
__forceinline__ __device__ float3 operator*(const float3& a, const float3& b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __device__ float3 operator*(const float3& a, const float s)
{
	return make_float3(a.x * s, a.y * s, a.z * s);
}
__forceinline__ __device__ float3 operator*(const float s, const float3& a)
{
	return make_float3(a.x * s, a.y * s, a.z * s);
}
__forceinline__ __device__ void operator*=(float3& a, const float3& s)
{
	a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
__forceinline__ __device__ void operator*=(float3& a, const float s)
{
	a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
__forceinline__ __device__ float3 operator/(const float3& a, const float3& b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__forceinline__ __device__ float3 operator/(const float3& a, const float s)
{
	float inv = 1.0f / s;
	return a * inv;
}
__forceinline__ __device__ float3 operator/(const float s, const float3& a)
{
	return make_float3(s / a.x, s / a.y, s / a.z);
}
__forceinline__ __device__ void operator/=(float3& a, const float s)
{
	float inv = 1.0f / s;
	a *= inv;
}
/** @} */

/** lerp */
__forceinline__ __device__ float3 lerp(const float3& a, const float3& b, const float t)
{
	return a + t * (b - a);
}

/** bilerp */
__forceinline__ __device__ float3 bilerp(const float3& x00, const float3& x10, const float3& x01, const float3& x11,
	const float u, const float v)
{
	return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
* @{
*/
__forceinline__ __device__ float3 clamp(const float3& v, const float a, const float b)
{
	return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

__forceinline__ __device__ float3 clamp(const float3& v, const float3& a, const float3& b)
{
	return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** dot product */
__forceinline__ __device__ float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** cross product */
__forceinline__ __device__ float3 cross(const float3& a, const float3& b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__forceinline__ __device__ void grad_cross(
	const float3 & a, const float3 & b,const float3 & grad_output,
	float3 & grad_a, float3 & grad_b
){
	grad_a = make_float3(
		-grad_output.y * b.z + grad_output.z * b.y ,
		grad_output.x * b.z - grad_output.z * b.x,
		-grad_output.x * b.y + grad_output.y * b.x
	);
	grad_b = make_float3(
		grad_output.y * a.z - grad_output.z * a.y,
		-grad_output.x * a.z + grad_output.z * a.x,
		grad_output.x * a.y - grad_output.y * a.x
	);
}

/** length */
__forceinline__ __device__ float length(const float3& v)
{
	return sqrtf(dot(v, v));
}

__forceinline__ __device__ float3 grad_length(const float3 & v, float grad) {
	float L = sqrtf(dot(v, v));
	return v * (grad / L);
}

/** normalize */
__forceinline__ __device__ float3 normalize(const float3& v)
{
	float invLen = 1.0f / sqrtf(dot(v, v));
	return v * invLen;
}

__forceinline__ __device__ float3 grad_normalize(const float3 & v, const float3 & grad_output)
{
	float v2 = dot(v, v);
	float v1_5 = sqrtf(v2) * v2 ; // (x^2+y^2+z^2)^(+3/2)
	float gx = grad_output.x * (v.y * v.y + v.z * v.z) - grad_output.y * (v.x * v.y) - grad_output.z * (v.x * v.z);
	float gy = -grad_output.x * (v.x * v.y) + grad_output.y * (v.x * v.x + v.z * v.z) - grad_output.z * (v.y * v.z);
	float gz = -grad_output.x * (v.x * v.z) - grad_output.y * (v.y * v.z) + grad_output.z * (v.x * v.x + v.y * v.y);
	return make_float3(gx,gy,gz) / v1_5;
}

/** floor */
__forceinline__ __device__ float3 floor(const float3& v)
{
	return make_float3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
}

/** reflect */
__forceinline__ __device__ float3 reflect(const float3& i, const float3& n)
{
	return i - 2.0f * n * dot(n, i);
}