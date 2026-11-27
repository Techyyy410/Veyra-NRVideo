// EdgeFill: fills the letterbox/pillarbox bars with real picture content that
// has scrolled off the picture's edges, tracked by per-edge camera motion.
//
// Every new source frame (compute, before the blit):
//   EdgeLuma    current frame -> half and quarter resolution luma
//   EdgeSearch  coarse: SAD over a wide window of integer shifts (quarter res)
//   EdgeSolve   coarse: pick the best shift per edge
//   EdgeSearch  refine: SAD over +-3 around it (half res)
//   EdgeSolve   refine: sub-pixel shift, reliability, scene cut, offsets
//   EdgeMask    learn screen-fixed / non-world pixels (HUD) from motion
//   EdgeUpdate  write the picture's edge bands into the world canvases
//   EdgeCopy    current luma becomes the previous luma
// Every present (graphics, after the blit): EdgeFillComposite draws the
// canvases into the bars over the ambient fill, faded by age.
//
// Units: the picture is normalised to PICTURE_H canvas texels high. Motion is
// the content displacement prev -> cur in canvas texels: cur(x) = prev(x - m).
// World = picture + offset, so a static world point keeps its world position
// when offset -= m. Canvases wrap (toroidal), offsets are kept in [0, size).

#ifndef EDGE_FILL_HLSLI
#define EDGE_FILL_HLSLI

cbuffer EdgeConstants : register(b0)
{
    uint4 ec0; // x halfW, y halfH, z quarterW, w quarterH
    uint4 ec1; // x canvasW, y canvasH, z pictureW, w pictureH (canvas texels)
    uint4 ec2; // x mode (search/solve: 0 coarse, 1 refine), y linear input, z band texels, w reset request
    uint4 ec3; // x write mask threshold (asfloat), y,z,w reserved
    float4 exclude0; // picture-uv rects (x0,y0,x1,y1) never written: fixed HUD such as
    float4 exclude1; // a minimap, which the learned mask cannot see where it is flat
};

bool Excluded(float2 uv)
{
    return (all(uv >= exclude0.xy) && all(uv < exclude0.zw)) ||
           (all(uv >= exclude1.xy) && all(uv < exclude1.zw));
}

Texture2D<float4> sourceTex : register(t0);
RWTexture2D<float>  curHalf    : register(u0);
RWTexture2D<float>  prevHalf   : register(u1);
RWTexture2D<float>  curQuarter : register(u2);
RWTexture2D<float>  prevQuarter: register(u3);
RWTexture2D<float>  hudMask    : register(u4);  // quarter res, 0 = world, 1 = screen-fixed / unreliable
RWTexture2D<float2> sadTex     : register(u5);  // (cost sum, weight sum) per candidate, row = edge
RWTexture2D<float4> stateTex   : register(u6);  // see STATE_* below
RWTexture2D<float4> canvasL    : register(u7);
RWTexture2D<float4> canvasR    : register(u8);
RWTexture2D<uint>   stampL     : register(u9);
RWTexture2D<uint>   stampR     : register(u10);

// State texels (RGBA32F, 8x1).
#define STATE_EDGE_L   0   // offset.xy, motion.xy (canvas texels)
#define STATE_EDGE_R   1
#define STATE_FRAME    2   // frame, validFrom L, validFrom R, flags
#define STATE_COARSE   3   // coarse best L.xy, R.xy (quarter texels, absolute shift)
#define STATE_COARSE_Q 4   // coarse cost L, ratio L, cost R, ratio R
#define STATE_ZERO     5   // zero-shift cost L, R, coarse on border L, R
#define STATE_TRACK    6   // unreliable run L, R, refine cost L, R
#define STATE_DEBUG    7

#define FLAG_DUPLICATE 1u
#define FLAG_RELIABLE_L 2u
#define FLAG_RELIABLE_R 4u
#define FLAG_CUT 8u

// Analysis strips (picture uv). The lower part is left out of the motion
// estimate: it is where games put minimaps, and it is mostly near ground.
static const float STRIP_U0 = 0.02;
static const float STRIP_U1 = 0.30;
static const float STRIP_V0 = 0.06;
static const float STRIP_V1 = 0.80;

// Coarse search, quarter-res texels around the previous motion (1 quarter
// texel = 4 canvas texels, so +-40 is +-160 px/frame on a 1080p picture).
static const int COARSE_RX = 40;
static const int COARSE_RY = 10;
static const int COARSE_W = 2 * COARSE_RX + 1;
static const int COARSE_H = 2 * COARSE_RY + 1;
static const int COARSE_COUNT = COARSE_W * COARSE_H; // + 1 zero-shift candidate
// Refine search, half-res texels around twice the coarse result.
static const int REFINE_R = 3;
static const int REFINE_W = 2 * REFINE_R + 1;
static const int REFINE_COUNT = REFINE_W * REFINE_W;

static const float DUPLICATE_COST = 0.006; // per-pixel mean |dL| of a repeated frame
static const float RELIABLE_RATIO = 0.6;   // best / mean cost must be below this
static const float RELIABLE_COST  = 0.10;
static const float CUT_COST       = 0.14;
static const int   UNRELIABLE_RUN = 8;
// Motion below this (canvas texels/frame) counts as still, so estimate jitter
// on a near-still camera does not walk the offsets and smear the edge.
static const float MOTION_DEADZONE = 0.35;     // frames without a lock before the history is dropped

uint StateFlags() { return uint(stateTex[uint2(STATE_FRAME, 0)].w + 0.5); }

float Luma(float3 rgb)
{
    float l = dot(max(rgb, 0), float3(0.2126, 0.7152, 0.0722));
    // Linear sources (scRGB, or SDR before the sRGB encode) get a perceptual
    // curve so dark detail counts and HDR highlights do not dominate the SAD.
    if (ec2.y != 0) l = sqrt(l / (1.0 + l) * 2.0);
    return saturate(l);
}

uint2 WrapCanvas(int2 p)
{
    const int2 size = int2(ec1.xy);
    return uint2(((p % size) + size) % size);
}

#endif
