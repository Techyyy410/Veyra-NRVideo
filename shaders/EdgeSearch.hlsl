// EdgeSearch: mean |cur(p) - prev(p - d)| over an edge strip for one candidate
// shift d per thread group. Group x = candidate, group z = edge (0 left, 1 right).
// Coarse (mode 0): quarter res, window around the previous motion, plus one
// extra candidate (index COARSE_COUNT) at zero shift for duplicate detection.
// Refine (mode 1): half res, +-REFINE_R around twice the coarse best shift.
#include "EdgeFill.hlsli"

groupshared float2 partial[256];

float LoadCur(uint refine, int2 p)  { return refine ? curHalf[p]  : curQuarter[p]; }
float LoadPrev(uint refine, int2 p) { return refine ? prevHalf[p] : prevQuarter[p]; }

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    const uint refine = ec2.x;
    const uint edge = group.z;
    const int candidate = int(group.x);
    int2 shift;
    if (refine) {
        if (candidate >= REFINE_COUNT) return;
        const float4 coarse = stateTex[uint2(STATE_COARSE, 0)];
        const int2 centre = int2(round(edge ? coarse.zw : coarse.xy)) * 2;
        shift = centre + int2(candidate % REFINE_W, candidate / REFINE_W) - REFINE_R;
    } else {
        if (candidate > COARSE_COUNT) return;
        if (candidate == COARSE_COUNT) shift = 0;
        else {
            // Predict from the last motion (canvas texels -> quarter texels).
            const float4 track = stateTex[uint2(edge ? STATE_EDGE_R : STATE_EDGE_L, 0)];
            const int2 centre = int2(round(track.zw / 4.0));
            shift = centre + int2(candidate % COARSE_W - COARSE_RX, candidate / COARSE_W - COARSE_RY);
        }
    }
    const int2 size = int2(refine ? ec0.xy : ec0.zw);
    const float u0 = edge ? 1.0 - STRIP_U1 : STRIP_U0;
    const float u1 = edge ? 1.0 - STRIP_U0 : STRIP_U1;
    const int2 lo = int2(float2(u0, STRIP_V0) * float2(size));
    const int2 hi = int2(float2(u1, STRIP_V1) * float2(size));
    const int2 extent = max((hi - lo + 1) / 2, 1); // every second texel
    const int count = extent.x * extent.y;
    const int maskScale = refine ? 2 : 1;
    float2 acc = 0;
    for (int i = int(thread); i < count; i += 256) {
        const int2 p = lo + int2(i % extent.x, i / extent.x) * 2;
        const int2 q = p - shift;
        if (any(q < 0) || any(q >= size)) continue;
        const float w = Excluded((float2(p) + 0.5) / float2(size)) ? 0.0 : 1.0 - saturate(hudMask[uint2(p / maskScale)]);
        acc += float2(abs(LoadCur(refine, p) - LoadPrev(refine, q)) * w, w);
    }
    partial[thread] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (thread < s) partial[thread] += partial[thread + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread == 0) sadTex[uint2(candidate, edge)] = partial[0];
}
