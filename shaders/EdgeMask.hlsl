// EdgeMask: learns which quarter-res pixels do not follow the world while the
// camera moves: screen-fixed HUD (matches the previous frame in place, not
// where the world went) and anything else that disagrees with the edge motion.
// It only learns while its edge has a reliable, non-trivial motion, so a still
// camera neither builds nor forgets anything.
#include "EdgeFill.hlsli"

float PrevAt(float2 p)
{
    const float2 size = float2(ec0.zw);
    p = clamp(p, 0, size - 1.001);
    const int2 b = int2(floor(p));
    const float2 f = p - float2(b);
    const float a = prevQuarter[b], c = prevQuarter[b + int2(1, 0)];
    const float d = prevQuarter[b + int2(0, 1)], e = prevQuarter[b + int2(1, 1)];
    return lerp(lerp(a, c, f.x), lerp(d, e, f.x), f.y);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint2 size = ec0.zw;
    if (any(id.xy >= size)) return;
    const uint flags = StateFlags();
    if (flags & (FLAG_DUPLICATE | FLAG_CUT)) return;
    const uint edge = id.x < size.x / 2 ? 0u : 1u;
    if (!(flags & (edge ? FLAG_RELIABLE_R : FLAG_RELIABLE_L))) return;
    const float2 motion = stateTex[uint2(edge ? STATE_EDGE_R : STATE_EDGE_L, 0)].zw / 4.0;
    if (length(motion) < 0.75) return;
    const int2 p = int2(id.xy);
    const float cur = curQuarter[p];
    const float2 from = float2(p) - motion;
    float moved = 1e9;
    [unroll] for (uint i = 0; i < 5; ++i) {
        const float2 jitter = i == 0 ? float2(0, 0) : i == 1 ? float2(0.5, 0) : i == 2 ? float2(-0.5, 0)
                            : i == 3 ? float2(0, 0.5) : float2(0, -0.5);
        moved = min(moved, abs(cur - PrevAt(from + jitter)));
    }
    const float still = abs(cur - prevQuarter[p]);
    const int2 last = int2(size) - 1;
    const float contrast = max(max(abs(cur - curQuarter[clamp(p + int2(1, 0), 0, last)]),
                                   abs(cur - curQuarter[clamp(p - int2(1, 0), 0, last)])),
                               max(abs(cur - curQuarter[clamp(p + int2(0, 1), 0, last)]),
                                   abs(cur - curQuarter[clamp(p - int2(0, 1), 0, last)])));
    if (contrast < 0.02 && moved < 0.02) return; // flat: no evidence either way
    float m = hudMask[p];
    if (moved > 0.05 && still < 0.5 * moved) m += 0.35;  // stayed put while the world moved
    else if (moved > 0.08) m += 0.15;                    // moves its own way
    else if (moved < 0.035) m -= 0.08;                   // follows the world
    hudMask[p] = saturate(m);
}
