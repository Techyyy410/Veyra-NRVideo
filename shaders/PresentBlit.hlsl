// PresentBlit: pixel-shader blit of the working frame onto a flip-model
// swapchain back buffer. Flip back buffers may only transition between
// PRESENT and RENDER_TARGET (not UAV), so the present path is a fullscreen
// triangle sample instead of a compute write.

#include "PresentColor.hlsli"
#include "PresentationRotation.hlsli"
Texture2D<float4> sourceTex : register(t0);

cbuffer PresentBlitConstants : register(b0)
{
    // 64 = 输出前加抖动（12 位源量化到 10 位交换链时抑制色带）。
    // 64 = dither before output (suppresses banding when a 12-bit source is quantised
    // onto a 10-bit swapchain).
    float4 srcDims; // z flags: 1=sRGB encode, 2=fine, 4=PQ encode, 8=SDR white, 16=HDR curve, 32=PQ, 64/128=rotation, 256=dither decode
    float4 dstDims; // x/y=client extent z/w=view center
    // Custom: static HDR brightness curve, applied on absolute nits before the PQ
    // encode. See veyra/engine/HdrOutputTuning.h for the contract; every field is
    // a no-op at its default (lift 0, mid 203, start 0, cap 0).
    float4 curve0;  // x=shadow lift EV, y=shadow range nits, z=mid-gray nits, w=highlight start nits
    float4 curve1;  // x=roll-off scale nits, y=peak cap nits (0 = off)
};

SamplerState linearClamp : register(s0);

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

float4 CubicWeights(float t)
{
    float t2=t*t,t3=t2*t;
    return float4(-0.5*t+t2-0.5*t3,1.0-2.5*t2+1.5*t3,
                  0.5*t+2.0*t2-1.5*t3,-0.5*t2+0.5*t3);
}

// Catmull-Rom (B = 0, C = 1/2), the same cubic the magnification path uses.
// Measured on real game frames (2026-11-18) it keeps more detail than the
// softer Mitchell filter while still removing the aliasing of the old blit.
float4 CatmullRomWeight4(float4 x)
{
    x=abs(x);
    const float4 inner=(1.5*x-2.5)*x*x+1.0;
    const float4 outer=((-0.5*x+2.5)*x-4.0)*x+2.0;
    return select(x<1.0,inner,select(x<2.0,outer,0.0));
}

// Up to 2:1 along one source axis the widened kernel covers at most eight
// texels: two in the left negative lobe, up to four in the positive lobe, two
// in the right negative lobe. Four taps, each pairing texels of one lobe, so
// one bilinear fetch blends each pair exactly. pos: tap positions in texels.
void MinifyAxis(float p,float scale,out float4 pos,out float4 weight)
{
    const float invScale=1.0/scale;
    const float first=floor(p-scale)+1.0;   // first texel of the positive lobe
    const float count=ceil(p+scale)-first;  // texels in it: 1..4
    const float4 start=float4(first-2.0,first,first+2.0,first+count);
    const float4 a=CatmullRomWeight4((start-p)*invScale)*float4(1.0,1.0,count>=3.0?1.0:0.0,1.0);
    const float4 b=CatmullRomWeight4((start+1.0-p)*invScale)*float4(1.0,count>=2.0?1.0:0.0,count>=4.0?1.0:0.0,1.0);
    weight=a+b;
    pos=start+select(weight!=0.0,b/weight,0.0);
}

// Anti-ringing for linear-light HDR: next to a highlight far above SDR white
// (scRGB output filters linear values, 1.0 = 80 nits) a negative lobe would
// draw a dark ring, so where any fetched sample exceeds 1.0 the result is held
// within the range of the positive-lobe samples. SDR-range pictures, and PQ
// surfaces whose code values never exceed 1.0, keep the full Catmull-Rom.
float4 HoldRinging(float4 value,float4 lo,float4 hi,float peak)
{
    return peak>1.0?clamp(value,lo,hi):value;
}

// 16 fetches for any shrink up to 2:1 (4K on a 1440p or 1080p screen).
float4 MinifySampleFast(float2 p,float2 scale,float2 size)
{
    float4 px,wx,py,wy;
    MinifyAxis(p.x,scale.x,px,wx);
    MinifyAxis(p.y,scale.y,py,wy);
    float4 sum=0,lo=1e30,hi=-1e30;float peak=0;
    [unroll]for(int j=0;j<4;++j){
        [unroll]for(int i=0;i<4;++i){
            const float w=wx[i]*wy[j];
            if(w!=0.0){
                const float4 c=sourceTex.SampleLevel(linearClamp,(float2(px[i],py[j])+0.5)/size,0);
                sum+=c*w;
                peak=max(peak,max(c.r,max(c.g,c.b)));
                if(i>=1&&i<=2&&j>=1&&j<=2){lo=min(lo,c);hi=max(hi,c);}
            }
        }
    }
    return HoldRinging(sum/(dot(wx,1.0)*dot(wy,1.0)),lo,hi,peak);
}

// 2:1 to 4:1 (a large frame in a small window): the same lobe pairing with
// up to four left, eight positive and four right texels, 64 fetches.
void MinifyAxis8(float p,float scale,out float4 pos0,out float4 w0,out float4 pos1,out float4 w1)
{
    const float invScale=1.0/scale;
    const float first=floor(p-scale)+1.0;
    const float count=ceil(p+scale)-first;  // positive-lobe texels: 1..8
    const float right=first+count;
    const float4 s0=float4(first-4.0,first-2.0,first,first+2.0);
    const float4 s1=float4(first+4.0,first+6.0,right,right+2.0);
    const float4 a0=CatmullRomWeight4((s0-p)*invScale)*float4(1.0,1.0,1.0,count>=3.0?1.0:0.0);
    const float4 b0=CatmullRomWeight4((s0+1.0-p)*invScale)*float4(1.0,1.0,count>=2.0?1.0:0.0,count>=4.0?1.0:0.0);
    const float4 a1=CatmullRomWeight4((s1-p)*invScale)*float4(count>=5.0?1.0:0.0,count>=7.0?1.0:0.0,1.0,1.0);
    const float4 b1=CatmullRomWeight4((s1+1.0-p)*invScale)*float4(count>=6.0?1.0:0.0,count>=8.0?1.0:0.0,1.0,1.0);
    w0=a0+b0;w1=a1+b1;
    pos0=s0+select(w0!=0.0,b0/w0,0.0);
    pos1=s1+select(w1!=0.0,b1/w1,0.0);
}

// One row of the 8x8 pattern: eight taps along x at one y position. Every tap
// is fetched (zero-weight ones add nothing), so the fetches issue together.
void MinifyRow8(float y,float wy,bool center,float4 px0,float4 px1,float4 wx0,float4 wx1,float2 size,
                inout float4 sum,inout float4 lo,inout float4 hi,inout float peak)
{
    [unroll]for(int i=0;i<8;++i){
        const float w=(i<4?wx0[i]:wx1[i-4])*wy;
        const float4 c=sourceTex.SampleLevel(linearClamp,(float2(i<4?px0[i]:px1[i-4],y)+0.5)/size,0);
        sum+=c*w;
        const bool used=w!=0.0;
        peak=used?max(peak,max(c.r,max(c.g,c.b))):peak;
        if(center&&i>=2&&i<=5&&used){lo=min(lo,c);hi=max(hi,c);}
    }
}

float4 MinifySampleFast8(float2 p,float2 scale,float2 size)
{
    float4 px0,wx0,px1,wx1,py0,wy0,py1,wy1;
    MinifyAxis8(p.x,scale.x,px0,wx0,px1,wx1);
    MinifyAxis8(p.y,scale.y,py0,wy0,py1,wy1);
    float4 sum=0,lo=1e30,hi=-1e30;float peak=0;
    [unroll]for(int j=0;j<4;++j)MinifyRow8(py0[j],wy0[j],j>=2,px0,px1,wx0,wx1,size,sum,lo,hi,peak);
    [unroll]for(int k=0;k<4;++k)MinifyRow8(py1[k],wy1[k],k<=1,px0,px1,wx0,wx1,size,sum,lo,hi,peak);
    return HoldRinging(sum/((dot(wx0,1.0)+dot(wx1,1.0))*(dot(wy0,1.0)+dot(wy1,1.0))),lo,hi,peak);
}

// Minification (2026-11-18): the bilinear sample this replaced skipped texels
// once the picture was shrunk (a 4K frame on a 1440p or 1080p screen, or in a
// window), which showed as jagged, shimmering, "pixelated" detail. scale: source
// texels per output pixel on each source axis (>= 1); the kernel is widened by
// it so detail finer than the screen can show is filtered instead of folded
// back. Weights are normalised and applied in the working frame's own encoding
// (sRGB code values for SDR output, linear for scRGB HDR, PQ for HDR10). Past
// 4:1 (a picture shown in a tiny window) taps that far apart miss the texture
// cache, so the 16-fetch pattern runs at its 2:1 width there: still filtered,
// at a bounded cost (RTX 5070: 4K into 640x360 0.19 ms instead of 0.7 ms).
float4 MinifySample(float2 p,float2 scale,float2 size)
{
    // Derivatives carry rounding, so an exact 2:1 can read 2.0000002.
    const float widest=max(scale.x,scale.y);
    if(widest<=2.001||widest>4.001)return MinifySampleFast(p,min(scale,2.0),size);
    return MinifySampleFast8(p,min(scale,4.0),size);
}

// Magnification and 1:1: a texel-exact copy when the grids line up, otherwise a
// Catmull-Rom cubic bounded to the local 2x2. Until 2026-11-18 only PS5 Remote
// Play presented through here; every other source was magnified bilinearly (soft).
float4 MagnifySample(float2 p,int2 last)
{
    if(all(abs(p-round(p))<0.0005))return sourceTex.Load(int3(clamp(int2(round(p)),int2(0,0),last),0));
    int2 base=int2(floor(p));float2 t=frac(p);
    float4 wx=CubicWeights(t.x),wy=CubicWeights(t.y);
    float4 result=0;
    [unroll]for(int y=0;y<4;++y){
        [unroll]for(int x=0;x<4;++x){
            int2 q=clamp(base+int2(x-1,y-1),int2(0,0),last);
            result+=sourceTex.Load(int3(q,0))*wx[x]*wy[y];
        }
    }
    // Bound cubic lobes to the local 2x2 neighborhood to avoid bright/dark
    // halos. Keep negative/HDR values when present in the actual neighborhood.
    float4 a=sourceTex.Load(int3(clamp(base,int2(0,0),last),0));
    float4 b=sourceTex.Load(int3(clamp(base+int2(1,0),int2(0,0),last),0));
    float4 c=sourceTex.Load(int3(clamp(base+int2(0,1),int2(0,0),last),0));
    float4 d=sourceTex.Load(int3(clamp(base+int2(1,1),int2(0,0),last),0));
    return clamp(result,min(min(a,b),min(c,d)),max(max(a,b),max(c,d)));
}

// ---------------------------------------------------------------------------
// MOD: ambient bar fill. Instead of black letterbox/pillarbox bars, the bars show
// a mirrored, heavily blurred, dimmed continuation of the picture's nearest edge
// (like YouTube's ambient mode). The picture itself is not touched. Tune here:
static const float AMBIENT_BLUR_NEAR = 0.012; // blur radius next to the picture (fraction of picture height)
static const float AMBIENT_BLUR_FAR  = 0.10;  // blur radius at the outer screen edge
static const float AMBIENT_DIM_NEAR  = 0.55;  // brightness next to the picture (1 = full)
static const float AMBIENT_DIM_FAR   = 0.10;  // brightness at the outer screen edge
static const int   AMBIENT_TAPS      = 128;   // blur samples per bar pixel (quality vs GPU cost)

// viewUv: display-space uv (picture = 0..1), outside the picture. Returns the
// blurred colour and, in dim, how much to darken it.
float4 AmbientFill(float2 viewUv,uint turns,uint2 pixel,out float dim)
{
    // How far the bars reach in uv on each side, so 0..1 across the bar.
    const float fit=min(dstDims.x/srcDims.x,dstDims.y/srcDims.y)*srcDims.w;
    const float2 k=dstDims.xy/(srcDims.xy*fit);
    const float2 lo=-.5*k+dstDims.zw,hi=.5*k+dstDims.zw;
    const float2 over=max(max(-viewUv,viewUv-1),0);
    const float2 extent=max(float2(viewUv.x<0?-lo.x:hi.x-1,viewUv.y<0?-lo.y:hi.y-1),1e-4);
    const float t=saturate(max(over.x/extent.x,over.y/extent.y));
    dim=lerp(AMBIENT_DIM_NEAR,AMBIENT_DIM_FAR,smoothstep(0,1,t));
    // Mirror across the picture edge so the bar starts with what is at the edge.
    float2 m=select(viewUv<0,-viewUv,select(viewUv>1,2-viewUv,viewUv));
    m=saturate(m);
    const float radius=lerp(AMBIENT_BLUR_NEAR,AMBIENT_BLUR_FAR,t);
    const float2 r2=float2(radius*srcDims.y/srcDims.x,radius); // round in screen space
    // Vogel spiral, rotated per pixel so undersampling reads as fine grain
    // instead of ghost copies of bright HUD elements.
    const float rot=6.2831853*frac(52.9829189*frac(dot(float2(pixel),float2(0.06711056,0.00583715))));
    float3 sum=0;
    [loop]for(int i=0;i<AMBIENT_TAPS;++i){
        float s,c;sincos(i*2.39996323+rot,s,c);
        const float2 o=float2(c,s)*sqrt((i+.5)/AMBIENT_TAPS)*r2;
        sum+=sourceTex.SampleLevel(linearClamp,PresentationSourceUv(saturate(m+o),turns),0).rgb;
    }
    return float4(sum/AMBIENT_TAPS,1);
}
// ---------------------------------------------------------------------------

float4 psMain(VSOut input) : SV_Target
{
    // zoom==0 retains the original contract for diagnostic harness callers.
    float2 uv=input.uv;
    bool outside=false;
    if(srcDims.w>0){
        float fit=min(dstDims.x/srcDims.x,dstDims.y/srcDims.y)*srcDims.w;
        uv=(input.uv-.5)*dstDims.xy/(srcDims.xy*fit)+dstDims.zw;
        outside=any(uv<0)||any(uv>1);
    }
    uint flags=uint(srcDims.z+0.5);
    const float2 viewUv=uv; // MOD: display-space uv, kept for the ambient fill
    uv=PresentationSourceUv(uv,(flags>>6)&3);
    // Texel position and its screen derivatives, taken before the letterbox exit
    // so every pixel of a quad still has them (derivatives in divergent flow are
    // undefined; the old cubic path measured them after it).
    uint width,height;sourceTex.GetDimensions(width,height);
    const float2 size=float2(width,height);
    const float2 p=uv*size-0.5;
    const float2 dx=ddx(p),dy=ddy(p);
    // MOD: the alpha round-trip path (256) keeps transparent bars.
    if(outside&&(flags&256))return float4(0,0,0,0);
    float4 color;
    float ambientDim=1;
    if(outside)color=AmbientFill(viewUv,(flags>>6)&3,uint2(input.pos.xy),ambientDim);
    else if(flags&2){
        // Source texels per output pixel along each source axis (a quarter turn
        // swaps which screen axis drives which).
        const float2 scale=max(abs(dx),abs(dy));
        const int2 last=int2(width,height)-1;
        color=all(scale<1.001)?MagnifySample(p,last):MinifySample(p,max(scale,1.0),size);
    }else color=sourceTex.SampleLevel(linearClamp,uv,0);
    return PresentFinish(color,flags,uint2(input.pos.xy),ambientDim,curve0,curve1);
}
