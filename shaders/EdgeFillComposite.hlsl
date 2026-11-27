// EdgeFillComposite: draws the edge history canvases (EdgeFill.hlsli) into the
// bars, alpha-blended over the ambient fill PresentBlit already drew there.
// Alpha is the history's confidence: full while fresh, fading with age, zero
// where nothing was ever seen. The colour goes through PresentBlit's exact
// finishing so the bars and the picture encode alike.
#include "PresentColor.hlsli"

cbuffer CompositeConstants : register(b0)
{
    // 0-15: PresentBlit's constants, unchanged.
    float4 srcDims;
    float4 dstDims;
    float4 curve0;
    float4 curve1;
    // 16-31: edge fill.
    float4 canvasDims; // x,y canvas size, z,w picture size (canvas texels)
    float4 timing;     // x hold frames, y fade frames, z strength, w unused
    float4 shading;    // x dim near, y dim far, z blur near, w blur far (canvas texels)
    float4 reserved;
};

Texture2D<float4> canvasL : register(t0);
Texture2D<float4> canvasR : register(t1);
Texture2D<uint>   stampL  : register(t2);
Texture2D<uint>   stampR  : register(t3);
Texture2D<float4> stateTex : register(t4);
SamplerState linearClamp : register(s0);
SamplerState linearWrap  : register(s1);

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOut vsMain(uint vertexId : SV_VertexID)
{
    VSOut o;
    const float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    o.uv = uv;
    return o;
}

static const int TAPS = 12;

float4 psMain(VSOut input) : SV_Target
{
    const uint flags = uint(srcDims.z + 0.5);
    if ((flags & 256) || srcDims.w <= 0 || ((flags >> 6) & 3) != 0) discard;
    const float fit = min(dstDims.x / srcDims.x, dstDims.y / srcDims.y) * srcDims.w;
    const float2 k = dstDims.xy / (srcDims.xy * fit);
    const float2 viewUv = (input.uv - .5) * k + dstDims.zw;
    // Only the side bars; the picture and any top/bottom bars are left alone.
    if (viewUv.x >= 0 && viewUv.x <= 1) discard;
    if (viewUv.y < 0 || viewUv.y > 1) discard;
    const uint edge = viewUv.x > 1 ? 1u : 0u;

    const float4 track = stateTex.Load(int3(edge, 0, 0));
    const float4 frameState = stateTex.Load(int3(2, 0, 0));
    const float frame = frameState.x;
    const float validFrom = edge ? frameState.z : frameState.y;
    if (frame < 1) discard;

    // How far across the bar, 0 at the picture, 1 at the screen edge.
    const float2 lo = -.5 * k + dstDims.zw, hi = .5 * k + dstDims.zw;
    const float extent = max(edge ? hi.x - 1 : -lo.x, 1e-4);
    const float t = saturate((edge ? viewUv.x - 1 : -viewUv.x) / extent);

    const float2 canvas = canvasDims.xy;
    const float2 world = viewUv * canvasDims.zw + track.xy; // canvas texels
    const float radius = lerp(shading.z, shading.w, t);
    // Per-pixel rotated spiral, like the ambient fill, so sparse taps read as grain.
    const float rot = 6.2831853 * frac(52.9829189 * frac(dot(input.pos.xy, float2(0.06711056, 0.00583715))));
    float3 sum = 0;
    float weight = 0, confidence = 0;
    [unroll] for (int i = 0; i < TAPS; ++i) {
        float s, c; sincos(i * 2.39996323 + rot, s, c);
        const float2 at = world + float2(c, s) * sqrt((i + .5) / TAPS) * radius;
        const int2 cell = int2(floor(at));
        const int2 wrapped = ((cell % int2(canvas)) + int2(canvas)) % int2(canvas);
        const uint stamp = edge ? stampR.Load(int3(wrapped, 0)) : stampL.Load(int3(wrapped, 0));
        float conf = 0;
        if (stamp > 0 && float(stamp) >= validFrom) {
            const float age = frame - float(stamp);
            conf = saturate(1 - (age - timing.x) / max(timing.y, 1));
        }
        if (conf > 0) {
            const float3 rgb = edge ? canvasR.SampleLevel(linearWrap, at / canvas, 0).rgb
                                    : canvasL.SampleLevel(linearWrap, at / canvas, 0).rgb;
            sum += rgb * conf;
            weight += conf;
        }
        confidence += conf;
    }
    confidence /= TAPS;
    if (weight <= 1e-4 || confidence <= 1e-3) discard;
    const float dim = lerp(shading.x, shading.y, smoothstep(0, 1, t));
    float4 color = PresentFinish(float4(sum / weight, 1), flags, uint2(input.pos.xy), dim, curve0, curve1);
    color.a = saturate(confidence * timing.z);
    return color;
}
