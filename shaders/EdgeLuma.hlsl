// EdgeLuma: current frame -> half-res and quarter-res luma (see EdgeFill.hlsli).
// One thread per quarter texel; it writes its four half texels and their mean.
#include "EdgeFill.hlsli"
SamplerState linearClamp : register(s1);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= ec0.zw)) return;
    const float2 halfSize = float2(ec0.xy);
    float sum = 0;
    [unroll] for (uint i = 0; i < 4; ++i) {
        const uint2 h = id.xy * 2 + uint2(i & 1, i >> 1);
        // The half texel centre in picture uv; bilinear averages the source
        // texels under it exactly when the source is twice the half size.
        const float2 uv = (float2(h) + 0.5) / halfSize;
        const float l = Luma(sourceTex.SampleLevel(linearClamp, uv, 0).rgb);
        if (all(h < ec0.xy)) curHalf[h] = l;
        sum += l;
    }
    curQuarter[id.xy] = sum * 0.25;
}
