// EdgeUpdate: writes the picture's outer bands into the two world canvases at
// the current world offsets, stamped with the frame number. Pixels the mask
// marks as screen-fixed or non-world are skipped, so the HUD never ends up in
// the bars. One thread per canvas texel of a band; group z = edge.
#include "EdgeFill.hlsli"
SamplerState linearClamp : register(s1);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint flags = StateFlags();
    if (flags & FLAG_DUPLICATE) return;
    const uint edge = id.z;
    if (!(flags & FLAG_CUT) && !(flags & (edge ? FLAG_RELIABLE_R : FLAG_RELIABLE_L))) return;
    const float2 picture = float2(ec1.zw);
    const float band = float(ec2.z);
    const float bandStart = edge ? picture.x - band : 0.0;
    const float4 track = stateTex[uint2(edge ? STATE_EDGE_R : STATE_EDGE_L, 0)];
    const int2 origin = int2(floor(track.xy + float2(bandStart, 0)));
    const int2 texel = origin + int2(id.xy);
    // Picture position of this canvas texel's centre.
    const float2 p = float2(texel) + 0.5 - track.xy;
    if (p.x < bandStart || p.x >= bandStart + band || p.y < 0 || p.y >= picture.y) return;
    const float2 uv = p / picture;
    if (Excluded(uv)) return;
    const int2 q = int2(uv * float2(ec0.zw));
    const int2 last = int2(ec0.zw) - 1;
    float hud = 0;
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x)
            hud = max(hud, hudMask[clamp(q + int2(x, y), 0, last)]);
    if (hud > asfloat(ec3.x)) return;
    const float4 color = float4(sourceTex.SampleLevel(linearClamp, uv, 0).rgb, 1);
    const uint frame = uint(stateTex[uint2(STATE_FRAME, 0)].x + 0.5);
    const uint2 at = WrapCanvas(texel);
    if (edge) { canvasR[at] = color; stampR[at] = frame; }
    else      { canvasL[at] = color; stampL[at] = frame; }
}
