// EdgeSolve: one group. Threads 0-127 handle the left edge, 128-255 the right.
// Coarse (mode 0): best integer shift, mean cost (for a reliability ratio),
// zero-shift cost (duplicate detection) and whether the best is on the border.
// Refine (mode 1): best shift, sub-pixel V-fit, then the per-frame decisions:
// duplicate, scene cut, reliability, and the world offsets.
#include "EdgeFill.hlsli"

groupshared float bestCost[256];
groupshared uint bestIndex[256];
groupshared float costSum[256];
groupshared float costCount[256];

float CostAt(int candidate, uint edge)
{
    const float2 s = sadTex[uint2(candidate, edge)];
    return s.y > 32.0 ? s.x / s.y : 1e9;
}

// Equiangular (V) fit, the right model for a sum of absolute differences.
float VFit(float minus, float centre, float plus)
{
    if (minus >= 1e8 || plus >= 1e8) return 0;
    const float d = minus > plus ? 0.5 * (minus - plus) / max(minus - centre, 1e-6)
                                 : 0.5 * (minus - plus) / max(plus - centre, 1e-6);
    return clamp(d, -0.5, 0.5);
}

float WrapOffset(float v, float size) { return v - floor(v / size) * size; }

[numthreads(256, 1, 1)]
void main(uint thread : SV_GroupIndex)
{
    const uint refine = ec2.x;
    const uint edge = thread >> 7;
    const uint lane = thread & 127;
    const int count = refine ? REFINE_COUNT : COARSE_COUNT;
    float localBest = 1e10; uint localIndex = 0; float localSum = 0, localCount = 0;
    for (int k = int(lane); k < count; k += 128) {
        const float c = CostAt(k, edge);
        if (c < localBest) { localBest = c; localIndex = uint(k); }
        if (c < 1e8) { localSum += c; localCount += 1; }
    }
    bestCost[thread] = localBest; bestIndex[thread] = localIndex;
    costSum[thread] = localSum; costCount[thread] = localCount;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 64; s > 0; s >>= 1) {
        if (lane < s) {
            const uint o = thread + s;
            if (bestCost[o] < bestCost[thread] ||
                (bestCost[o] == bestCost[thread] && bestIndex[o] < bestIndex[thread])) {
                bestCost[thread] = bestCost[o]; bestIndex[thread] = bestIndex[o];
            }
            costSum[thread] += costSum[o]; costCount[thread] += costCount[o];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread != 0) return;

    if (!refine) {
        float4 coarse = 0, quality = 0, zero = 0;
        [unroll] for (uint e = 0; e < 2; ++e) {
            const uint root = e * 128;
            const int index = int(bestIndex[root]);
            const float4 track = stateTex[uint2(e ? STATE_EDGE_R : STATE_EDGE_L, 0)];
            const int2 centre = int2(round(track.zw / 4.0));
            const int2 cell = int2(index % COARSE_W, index / COARSE_W);
            const float2 shift = float2(centre + cell - int2(COARSE_RX, COARSE_RY));
            const float mean = costCount[root] > 0 ? costSum[root] / costCount[root] : 1e9;
            const bool border = cell.x == 0 || cell.y == 0 || cell.x == COARSE_W - 1 || cell.y == COARSE_H - 1;
            if (e) { coarse.zw = shift; quality.zw = float2(bestCost[root], bestCost[root] / max(mean, 1e-6)); zero.y = CostAt(COARSE_COUNT, 1); zero.w = border; }
            else   { coarse.xy = shift; quality.xy = float2(bestCost[root], bestCost[root] / max(mean, 1e-6)); zero.x = CostAt(COARSE_COUNT, 0); zero.z = border; }
        }
        stateTex[uint2(STATE_COARSE, 0)] = coarse;
        stateTex[uint2(STATE_COARSE_Q, 0)] = quality;
        stateTex[uint2(STATE_ZERO, 0)] = zero;
        return;
    }

    float4 frameState = stateTex[uint2(STATE_FRAME, 0)];
    const float4 quality = stateTex[uint2(STATE_COARSE_Q, 0)];
    const float4 zero = stateTex[uint2(STATE_ZERO, 0)];
    const float4 coarse = stateTex[uint2(STATE_COARSE, 0)];
    float4 trackState = stateTex[uint2(STATE_TRACK, 0)];
    const bool reset = ec2.w != 0;
    const bool duplicate = zero.x < DUPLICATE_COST && zero.y < DUPLICATE_COST;
    if (duplicate && !reset) {
        frameState.w = float(FLAG_DUPLICATE);
        stateTex[uint2(STATE_FRAME, 0)] = frameState;
        return;
    }
    const float frame = frameState.x + 1.0;
    const bool cut = reset || (quality.x > CUT_COST && quality.z > CUT_COST);
    uint flags = cut ? FLAG_CUT : 0u;
    const float2 canvasSize = float2(ec1.xy);
    [unroll] for (uint e = 0; e < 2; ++e) {
        const uint root = e * 128;
        float4 track = stateTex[uint2(e ? STATE_EDGE_R : STATE_EDGE_L, 0)];
        const int index = int(bestIndex[root]);
        const int2 cell = int2(index % REFINE_W, index / REFINE_W);
        const float centreCost = bestCost[root];
        float2 sub = 0;
        if (cell.x > 0 && cell.x < REFINE_W - 1)
            sub.x = VFit(CostAt(index - 1, e), centreCost, CostAt(index + 1, e));
        if (cell.y > 0 && cell.y < REFINE_W - 1)
            sub.y = VFit(CostAt(index - REFINE_W, e), centreCost, CostAt(index + REFINE_W, e));
        const int2 centre = int2(round(e ? coarse.zw : coarse.xy)) * 2;
        const float2 shiftHalf = float2(centre + cell - REFINE_R) + sub;
        const float ratio = e ? quality.w : quality.y;
        const bool border = (e ? zero.w : zero.z) > 0.5;
        const bool reliable = !cut && !border && ratio < RELIABLE_RATIO && centreCost < RELIABLE_COST;
        float run = e ? trackState.y : trackState.x;
        float valid = e ? frameState.z : frameState.y;
        if (cut) {
            track.zw = 0; run = 0; valid = frame;
        } else if (reliable) {
            track.zw = shiftHalf * 2.0;           // half texel = 2 canvas texels
            if (length(track.zw) < MOTION_DEADZONE) track.zw = 0;
            track.xy = float2(WrapOffset(track.x - track.z, canvasSize.x),
                              WrapOffset(track.y - track.w, canvasSize.y));
            run = 0;
            flags |= e ? FLAG_RELIABLE_R : FLAG_RELIABLE_L;
        } else {
            track.zw *= 0.5;                      // let the prediction relax towards still
            run += 1;
            if (run > UNRELIABLE_RUN) valid = frame;
        }
        stateTex[uint2(e ? STATE_EDGE_R : STATE_EDGE_L, 0)] = track;
        if (e) { trackState.y = run; trackState.w = centreCost; frameState.z = valid; }
        else   { trackState.x = run; trackState.z = centreCost; frameState.y = valid; }
    }
    frameState.x = frame;
    frameState.w = float(flags);
    stateTex[uint2(STATE_FRAME, 0)] = frameState;
    stateTex[uint2(STATE_TRACK, 0)] = trackState;
}
