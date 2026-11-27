// EdgeCopy: the current luma becomes the previous luma, unless this frame was
// a repeat (then the older frame stays the reference for the next real one).
#include "EdgeFill.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= ec0.zw)) return;
    if (StateFlags() & FLAG_DUPLICATE) return;
    prevQuarter[id.xy] = curQuarter[id.xy];
    [unroll] for (uint i = 0; i < 4; ++i) {
        const uint2 h = id.xy * 2 + uint2(i & 1, i >> 1);
        if (all(h < ec0.xy)) prevHalf[h] = curHalf[h];
    }
}
