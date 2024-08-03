#include <ATen/ATen.h>
#include <torch/script.h>

#include <vector>
#include <unordered_set>
#include <exception>


//////////////////////////////////
// K-nearest neighbor

#include "vec_math.h"
#include "nanoflann.hpp"

struct Point2D{
    float x,y;
};

template<typename scalar_t, int n_dims>
struct PointCloud2
{
    const scalar_t * accessor;
    size_t point_number;
    inline size_t kdtree_get_point_count() const { return point_number; }
    inline scalar_t kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		return  accessor[idx * n_dims + dim];
	}
    template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const
	{
		return false;
	}
};

template<typename scalar_t>
struct PointCloud
{
	torch::TensorAccessor<scalar_t, 2> * accessor; // 2 dim
	size_t point_number;
	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return point_number; }
	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate
	// value, the
	//  "if/else's" are actually solved at compile time.
	inline scalar_t kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		return  (*accessor)[idx][dim];
	}
	// Optional bounding-box computation: return false to default to a standard
	// bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned
	//   in "bb" so it can be avoided to redo it again. Look at bb.size() to
	//   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const
	{
		return false;
	}
};

float3 closesPointOnTriangle(
    const float3 & x0,
    const float3 & x1,
    const float3 & x2,
    const float3 & sourcePosition,
    float * oS, float * oT
){
    float3 edge0 = x1 - x0;
    float3 edge1 = x2 - x0;
    float3 v0 = x0 - sourcePosition;

    float a = dot(edge0, edge0);
    float b = dot(edge0, edge1);
    float c = dot(edge1, edge1);
    float d = dot(edge0, v0);
    float e = dot(edge1, v0);

    float det = a * c - b * b;
    float s = b * e - c * d;
    float t = b * d - a * e;

    if (s + t < det)
    {
        if (s < 0.f)
        {
            if (t < 0.f)
            {
                if (d < 0.f)
                {
                    s = clamp(-d / a, 0.f, 1.f);
                    t = 0.f;
                }
                else
                {
                    s = 0.f;
                    t = clamp(-e / c, 0.f, 1.f);
                }
            }
            else
            {
                s = 0.f;
                t = clamp(-e / c, 0.f, 1.f);
            }
        }
        else if (t < 0.f)
        {
            s = clamp(-d / a, 0.f, 1.f);
            t = 0.f;
        }
        else
        {
            float invDet = 1.f / det;
            s *= invDet;
            t *= invDet;
        }
    }
    else
    {
        if (s < 0.f)
        {
            float tmp0 = b + d;
            float tmp1 = c + e;
            if (tmp1 > tmp0)
            {
                float numer = tmp1 - tmp0;
                float denom = a - 2 * b + c;
                s = clamp(numer / denom, 0.f, 1.f);
                t = 1 - s;
            }
            else
            {
                t = clamp(-e / c, 0.f, 1.f);
                s = 0.f;
            }
        }
        else if (t < 0.f)
        {
            if (a + d > b + e)
            {
                float numer = c + e - b - d;
                float denom = a - 2 * b + c;
                s = clamp(numer / denom, 0.f, 1.f);
                t = 1 - s;
            }
            else
            {
                s = clamp(-e / c, 0.f, 1.f);
                t = 0.f;
            }
        }
        else
        {
            float numer = c + e - b - d;
            float denom = a - 2 * b + c;
            s = clamp(numer / denom, 0.f, 1.f);
            t = 1.f - s;
        }
    }

    *oS = s;
    *oT = t;
    return x0 + s * edge0 + t * edge1;
}




torch::Tensor getNearestTriangleID(
    const torch::Tensor& in_points, // Nx3
    const torch::Tensor& faces, // Fx3
    const torch::Tensor& query_points, // Mx3
    int32_t findNumber
){
    int64_t dim = in_points.dim();
	TORCH_CHECK(dim == 2, "input dimension must be 2, in format Nx3, x y z");
	TORCH_CHECK(in_points.size(1) == 3, "in_points must be Nx3");
	int64_t num_points = in_points.size(0);

    dim = faces.dim();
    TORCH_CHECK(dim == 2, "input dimension must be 2, in format Fx3");
    int64_t F = faces.size(0);
    TORCH_CHECK(faces.size(1) == 3);

    dim = query_points.dim();
	TORCH_CHECK(dim == 2, "input dimension must be 2, in format Mx3, x y z");
	TORCH_CHECK(query_points.size(1) == 3, "query_points must be Mx3");
	int64_t query_num_points = query_points.size(0);

	//at::TensorOptions opt(point_feature.dtype());
    at::TensorOptions opt_i(at::kInt);
    torch::Tensor out_face_index = torch::empty({query_num_points}, opt_i);

    auto ptr_in_points = in_points.accessor<float, 2>();
	auto ptr_query_points = query_points.accessor<float, 2>();
	//auto ptr_point_feature = point_feature.accessor<float, 2>();
	//auto ptr_out_point_feature = out_point_feature.accessor<float,2>();
	auto ptr_out_face_index = out_face_index.accessor<int,1>();
	auto ptr_faces = faces.accessor<int, 2>();

	////////////////////
    // process triangles
    std::vector<std::vector<int>> vertex_to_face;
    vertex_to_face.resize(num_points);

    for (int i = 0; i < F; i++) {
        for(int j = 0; j < 3; j++){
            vertex_to_face[ptr_faces[i][j]].push_back(i);
        }
    }
    ////////////////////
	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>, PointCloud<float>, 3> my_kdtree_t;
	PointCloud<float> cloud;

	cloud.accessor = &ptr_in_points;
	cloud.point_number = num_points;

    my_kdtree_t index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    std::vector<size_t> ret_index(findNumber);
    std::vector<float> out_dist_sqr(findNumber);
    for(auto i_q = 0; i_q < query_num_points; i_q++){
        nanoflann::KNNResultSet<float> resultSet(findNumber);
        resultSet.init(ret_index.data(), out_dist_sqr.data());

		if (!index.findNeighbors(resultSet, &ptr_query_points[i_q][0], nanoflann::SearchParams(10))) {
			throw std::runtime_error("KNN findNeighbors fail!\n");
		}
		// go through all faces and get the nearest dist to triangles
		std::unordered_set<int> touched_face;
		for (int i = 0; i < findNumber; i++) {
			int vertex_id = ret_index[i];
			std::vector<int> & faces = vertex_to_face[vertex_id];
			for (int j = 0; j < faces.size(); j++) {
				touched_face.insert(faces[j]);
			}
		}
		float closest_dist = 1e12;
		int sel_face = -1;
		//float sel_s, sel_t;
        for (auto it = touched_face.begin(); it != touched_face.end(); it++) {
		    int id = ptr_faces[*it][0];
		    float3 v0 = make_float3(
		        ptr_in_points[id][0], ptr_in_points[id][1], ptr_in_points[id][2]
		    );
		    id = ptr_faces[*it][1];
		    float3 v1 = make_float3(
		        ptr_in_points[id][0], ptr_in_points[id][1], ptr_in_points[id][2]
		    );
		    id = ptr_faces[*it][2];
		    float3 v2 = make_float3(
		        ptr_in_points[id][0], ptr_in_points[id][1], ptr_in_points[id][2]
		    );
		    float3 query_point = make_float3(
		        ptr_query_points[i_q][0], ptr_query_points[i_q][1], ptr_query_points[i_q][2]
		    );
			float s, t;
			float3 close_p = closesPointOnTriangle(
				v0,v1,v2, query_point,
				&s,&t
			);
			float3 diff = close_p - query_point;
			float dist2 = dot(diff,diff);
			if (dist2 < closest_dist){
				//close_p_ = close_p;
				sel_face = *it;
				//sel_s = s; sel_t = t;
				closest_dist = dist2;
			}
		}
		if(sel_face == -1){
		    throw std::runtime_error("no valid face find!\n");
		}
		ptr_out_face_index[i_q] = sel_face;
	}
	return out_face_index;
}


torch::Tensor getNearestFeature(
    const torch::Tensor& in_points, // Nx3
    const torch::Tensor& point_feature, // NxC
    const torch::Tensor& faces, // Fx3, int32
    const torch::Tensor& query_points, // Mx3
    int32_t findNumber
){

    int64_t dim = in_points.dim();
	TORCH_CHECK(dim == 2, "input dimension must be 2, in format Nx3, x y z");
	TORCH_CHECK(in_points.size(1) == 3, "in_points must be Nx3");
	int64_t num_points = in_points.size(0);

	dim = point_feature.dim();
	TORCH_CHECK(dim == 2, "input dimension must be 2, in format NxC");
	int64_t C = point_feature.size(1);
	TORCH_CHECK(point_feature.size(0) == num_points);

    dim = faces.dim();
    TORCH_CHECK(dim == 2, "input dimension must be 2, in format Fx3");
    int64_t F = faces.size(0);
    TORCH_CHECK(faces.size(1) == 3);

    dim = query_points.dim();
	TORCH_CHECK(dim == 2, "input dimension must be 2, in format Mx3, x y z");
	TORCH_CHECK(query_points.size(1) == 3, "query_points must be Mx3");
	int64_t query_num_points = query_points.size(0);

	at::TensorOptions opt(point_feature.dtype());
    torch::Tensor out_point_feature = torch::empty({query_num_points, C}, opt);

	auto ptr_in_points = in_points.accessor<float, 2>();
	auto ptr_query_points = query_points.accessor<float, 2>();
	auto ptr_point_feature = point_feature.accessor<float, 2>();
	auto ptr_out_point_feature = out_point_feature.accessor<float,2>();
	auto ptr_faces = faces.accessor<int, 2>();


    ////////////////////
    // process triangles
    std::vector<std::vector<int>> vertex_to_face;
    vertex_to_face.resize(num_points);

    for (int i = 0; i < F; i++) {
        for(int j = 0; j < 3; j++){
            vertex_to_face[ptr_faces[i][j]].push_back(i);
        }
    }

	////////////////////
	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>, PointCloud<float>, 3> my_kdtree_t;
	PointCloud<float> cloud;

	cloud.accessor = &ptr_in_points;
	cloud.point_number = num_points;

    my_kdtree_t index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    std::vector<size_t> ret_index(findNumber);
    std::vector<float> out_dist_sqr(findNumber);
    for(auto i_q = 0; i_q < query_num_points; i_q++){
        nanoflann::KNNResultSet<float> resultSet(findNumber);
        resultSet.init(ret_index.data(), out_dist_sqr.data());

		if (!index.findNeighbors(resultSet, &ptr_query_points[i_q][0], nanoflann::SearchParams(10))) {
			throw std::runtime_error("KNN findNeighbors fail!\n");
		}
		// go through all faces and get the nearest dist to triangles
		std::unordered_set<int> touched_face;
		for (int i = 0; i < findNumber; i++) {
			int vertex_id = ret_index[i];
			std::vector<int> & faces = vertex_to_face[vertex_id];
			for (int j = 0; j < faces.size(); j++) {
				touched_face.insert(faces[j]);
			}
		}
		float closest_dist = 1e12;
		int sel_face = -1;
		float sel_s, sel_t;
		for (auto it = touched_face.begin(); it != touched_face.end(); it++) {
		    int id = ptr_faces[*it][0];
		    float3 v0 = make_float3(
		        ptr_in_points[id][0], ptr_in_points[id][1], ptr_in_points[id][2]
		    );
		    id = ptr_faces[*it][1];
		    float3 v1 = make_float3(
		        ptr_in_points[id][0], ptr_in_points[id][1], ptr_in_points[id][2]
		    );
		    id = ptr_faces[*it][2];
		    float3 v2 = make_float3(
		        ptr_in_points[id][0], ptr_in_points[id][1], ptr_in_points[id][2]
		    );
		    float3 query_point = make_float3(
		        ptr_query_points[i_q][0], ptr_query_points[i_q][1], ptr_query_points[i_q][2]
		    );
			float s, t;
			float3 close_p = closesPointOnTriangle(
				v0,v1,v2, query_point,
				&s,&t
			);
			float3 diff = close_p - query_point;
			float dist2 = dot(diff,diff);
			if (dist2 < closest_dist){
				//close_p_ = close_p;
				sel_face = *it;
				sel_s = s; sel_t = t;
				closest_dist = dist2;
			}
		}
		if(sel_face == -1){
		    throw std::runtime_error("no valid face find!\n");
		}
		int id0 = ptr_faces[sel_face][0];
		int id1 = ptr_faces[sel_face][1];
		int id2 = ptr_faces[sel_face][2];
		for(int iC = 0; iC < C; iC++){
            ptr_out_point_feature[i_q][iC] = ptr_point_feature[id0][iC] * (1-sel_s-sel_t) +
                ptr_point_feature[id1][iC] * sel_s + ptr_point_feature[id2][iC] * sel_t;
		}
    }
    return out_point_feature;
}

