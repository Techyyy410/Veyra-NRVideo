// PresentColor: the present blit's colour finishing, shared with the bar
// composite (EdgeFillComposite) so both encode exactly alike.
#ifndef PRESENT_COLOR_HLSLI
#define PRESENT_COLOR_HLSLI
#include "HdrColor.hlsli"

// Lift the shadows without touching the highlights, move the mid-gray reference
// with the peak held fixed, then roll off and cap the top end.
float HdrCurveLuminance(float nits, float4 c0, float4 c1)
{
    if (c0.x != 0.0 && c0.y > 0.0) {
        const float w = saturate(1.0 - nits / c0.y);
        nits *= 1.0 + w * w * (exp2(c0.x) - 1.0);
    }
    if (c0.z > 0.0 && abs(c0.z - 203.0) > 0.5) {
        const float peak = c1.y > 0.0 ? c1.y : 1000.0;
        if (peak > c0.z + 1.0) {
            const float g = log(c0.z / peak) / log(203.0 / peak);
            nits = peak * pow(max(nits / peak, 0.0), g);
        }
    }
    if (c0.w > 0.0 && c1.x > 0.0 && nits > c0.w) nits = c0.w + (nits - c0.w) / (1.0 + (nits - c0.w) / c1.x);
    if (c1.y > 0.0) nits = min(nits, c1.y);
    return nits;
}

float3 HdrCurve(float3 nits, float4 c0, float4 c1)
{
    // A per-channel gamma/knee changes hue and clipping negative scRGB removes
    // BT.2020 colors outside BT.709. Apply one luminance gain to all components.
    const float y = dot(nits, float3(.212639, .715169, .072192));
    if (y <= 1e-6) return nits;
    float3 mapped = nits * (HdrCurveLuminance(y, c0, c1) / y);
    if (c1.y > 0.0) {
        const float3 rgb2020 = HdrTo2020(mapped);
        const float peak = max(max(rgb2020.r, rgb2020.g), rgb2020.b);
        if (peak > c1.y) mapped *= c1.y / peak;
    }
    return mapped;
}

// Ordered dither for the 12 -> 10 bit quantisation.
//
// The reconstruction produces 12 bits and the swapchain takes 10 (R10G10B10A2_UNORM).
// A 16-float texture is written by the shader and the hardware truncates it to 10 bits
// when the swapchain is resolved, so a smooth gradient that crosses one 10-bit step
// shows as a visible band. Adding a sub-LSB offset before that truncation turns the
// step into noise the eye averages away.
//
// The pattern is an 8x8 interleaved gradient noise (Jimenez 2014): cheap, no texture,
// and its visual character is stable under motion - a random offset would crawl.
// The amplitude is exactly one 10-bit step (1/1023) scaled to the shader's output
// domain, so the dither never moves a value by more than half a step either way.
float DitherOffset(uint2 pixel)
{
    float noise = frac(52.9829189 * frac(dot(float2(pixel), float2(0.06711056, 0.00583715))));
    return (noise - 0.5) / 1023.0;
}

// Everything after sampling: alpha round trip, PQ decode, SDR white in HDR,
// the HDR curve, a brightness scale (1 for the picture, the bar fills dim),
// the transfer encode and the dither. Flags as in PresentBlit's srcDims.z.
float4 PresentFinish(float4 color,uint flags,uint2 pixel,float dim,float4 curve0,float4 curve1)
{
    if((flags&256)&&color.a>1e-6)color.rgb/=color.a;
    if(flags&32)color.rgb=HdrDecodePq(color.rgb);
    if(flags&8)color.rgb*=203.0/80.0; // SDR comparison white in an HDR output.
    // Custom: static HDR curve, on absolute nits (the working space is linear
    // scRGB, 1.0 = 80 nits), before the transfer function is encoded.
    if(flags&16)color.rgb=HdrCurve(color.rgb*80.0,curve0,curve1)/80.0;
    color.rgb*=dim;
    if(flags&1){float3 c=max(color.rgb,0);color.rgb=lerp(1.055*pow(c,1.0/2.4)-0.055,c*12.92,step(c,0.0031308));}
    if(flags&4)color.rgb=HdrEncodePq(color.rgb);
    if(flags&256)color.rgb*=color.a;
    // Custom dither (moved off 256, which now carries the alpha round trip): the
    // offset belongs after the transfer encode, where one 10-bit step is exact.
    if(flags&512){
        color.rgb=saturate(color.rgb+DitherOffset(pixel));
    }
    return color;
}
#endif
