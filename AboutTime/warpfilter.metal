#include <metal_stdlib>

#include <CoreImage/CoreImage.h>

float2 to_normal(float2 coords, float width) {
    return (coords / width - 0.5) * 2.0; // [0..224] -> [-1..1]
}

float2 from_normal(float2 coords, float width) {
    return (coords / 2.0 + 0.5) * width;
}

float3 to_homogenous(float2 coords) {
    return float3(coords.x, coords.y, 1.0);
}

float2 from_homogenous(float3 coords) {
    return float2(coords.x/coords.z, coords.y/coords.z);
}

extern "C" {
    namespace coreimage {
        float2 warp(float3x3 transform, float width, destination dest) {
            float2 coord = dest.coord();
            coord.y = width - coord.y; //flip the image to apply transform
            float3 norm_homogenous = to_homogenous(to_normal(coord, width));
            float3 transformed_norm_homogenous = transform * norm_homogenous;
            float2 transformed_coord = from_normal(from_homogenous(transformed_norm_homogenous), width);
            transformed_coord.y = width - transformed_coord.y; //flip the image back
            return transformed_coord;
        }
    }
}
//   0    1    2    3    4    5    6    7    8    9   10   11
// x00, y00, x01, y01, x02, y02, x03, y03, x04, y04, x05, y05 ...

//[-1...1] /2 -> [-0.5...0.5]  +0.5 -> [0...1]

//[0...1] *2  -> [0...2]  -1 -> [-1...1]
