#pragma once
// EdgeFill: fills the side bars with picture content that scrolled off the
// picture's edges, tracked by per-edge camera motion, over the ambient fill
// PresentBlit draws there. The algorithm and its passes are described in
// shaders/EdgeFill.hlsli. This class only owns the GPU objects and records the
// passes into the presentation list.
//
// Settings: <shaders>/EdgeFill.ini (key=value, read on every source open), or
// VEYRA_EDGE_FILL=0 to switch it off. Any failure switches it off for the
// session; the ambient fill underneath keeps working on its own.
#include "veyra/Log.h"
#include "veyra/RuntimePaths.h"
#include "veyra/pipeline/GpuPassUtils.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <string>
#include <vector>

namespace veyra::engine {

class EdgeFill {
public:
    struct Settings {
        bool enabled=true;
        float strength=1.0f;      // history opacity over the ambient fill
        float holdFrames=90;      // source frames at full opacity after last seen
        float fadeFrames=150;     // then fades to the ambient fill over this many
        float dimNear=0.9f;       // brightness next to the picture
        float dimFar=0.5f;        // brightness at the screen edge
        float blurNear=0.5f;      // blur radius next to the picture (canvas texels, 1080p scale)
        float blurFar=7.0f;       // blur radius at the screen edge
        float band=0.30f;         // picture width fraction written per edge
        float maskThreshold=0.35f;
        // Picture-uv rects (x0,y0,x1,y1) never copied into the bars. The
        // default covers a bottom-left minimap (GTA). 0,0,0,0 = none.
        std::array<float,4> exclude0{0.0f,0.78f,0.20f,1.0f};
        std::array<float,4> exclude1{0.0f,0.0f,0.0f,0.0f};
    };

    // VideoPresenter::open: (re)binds the two real-frame resources. Pipelines
    // are kept across opens on the same device and output format.
    void open(ID3D12Device* device,ID3D12Resource* frame0,ID3D12Resource* frame1,
              DXGI_FORMAT frameFormat,unsigned workWidth,unsigned workHeight) {
        settings_=loadSettings();
        resetPending_=true;
        if(!settings_.enabled||failed_||!device||!frame0||!frame1||!workWidth||!workHeight){ready_=false;return;}
        if(device!=device_||frameFormat!=format_){
            release();device_=device;format_=frameFormat;
            if(!createPasses()){fail("pipeline creation");return;}
        }
        const unsigned pictureW=std::max(64u,unsigned(std::lround(kPictureH*double(workWidth)/workHeight/4.0))*4u);
        if(pictureW!=pictureW_){
            pictureW_=pictureW;
            if(!createResources()){fail("resource allocation");return;}
            log::info("edge-fill",std::format("canvas {}x{} picture {}x{} analysis {}x{}",
                kCanvasW,kCanvasH,pictureW_,kPictureH,pictureW_/2,kPictureH/2));
        }
        const auto heap=heap_->GetCPUDescriptorHandleForHeapStart();
        pipeline::makeSrv(device_,frame0,format_,{heap.ptr+0*increment_});
        pipeline::makeSrv(device_,frame1,format_,{heap.ptr+1*increment_});
        ready_=true;
    }
    void release() {
        ready_=false;device_=nullptr;format_=DXGI_FORMAT_UNKNOWN;pictureW_=0;
        for(auto* p:{&luma_,&search_,&solve_,&mask_,&update_,&copy_})*p={};
        composite_={};compositeFormat_=DXGI_FORMAT_UNKNOWN;heap_.Reset();
        for(auto& r:textures_)r.Reset();
    }
    bool ready() const {return ready_;}

    // A new real source frame: motion, mask and canvas update. `slot` is the
    // video frame slot (0 or 1); its resource must be in COMMON and is left
    // there. reset drops the history (the bars were not shown for a while).
    void update(ID3D12GraphicsCommandList* list,unsigned slot,ID3D12Resource* source,bool linearInput,bool reset) {
        if(!ready_||slot>1||!source)return;
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition.pResource=source;
        b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore=D3D12_RESOURCE_STATE_COMMON;b.Transition.StateAfter=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list->ResourceBarrier(1,&b);
        ID3D12DescriptorHeap* heaps[]={heap_.Get()};list->SetDescriptorHeaps(1,heaps);
        const unsigned qW=pictureW_/4,qH=kPictureH/4;
        const unsigned bandTexels=unsigned(std::clamp(settings_.band,0.05f,0.5f)*pictureW_);
        constants_={pictureW_/2,kPictureH/2,qW,qH, kCanvasW,kCanvasH,pictureW_,kPictureH,
                    0,linearInput?1u:0u,bandTexels,(reset||resetPending_)?1u:0u,
                    std::bit_cast<uint32_t>(settings_.maskThreshold),0,0,0};
        for(unsigned i=0;i<4;++i){
            constants_[16+i]=std::bit_cast<uint32_t>(settings_.exclude0[i]);
            constants_[20+i]=std::bit_cast<uint32_t>(settings_.exclude1[i]);
        }
        resetPending_=false;
        const auto run=[&](const pipeline::ComputePass& pass,unsigned mode,unsigned x,unsigned y,unsigned z){
            constants_[8]=mode;
            list->SetComputeRootSignature(pass.rootSig.Get());list->SetPipelineState(pass.pso.Get());
            list->SetComputeRoot32BitConstants(0,kConstants,constants_.data(),0);
            const auto gpu=heap_->GetGPUDescriptorHandleForHeapStart().ptr;
            list->SetComputeRootDescriptorTable(1,{gpu+size_t(slot)*increment_});
            list->SetComputeRootDescriptorTable(2,{gpu+size_t(kSrvSlots)*increment_});
            list->Dispatch(x,y,z);
            D3D12_RESOURCE_BARRIER uav{};uav.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;uav.UAV.pResource=nullptr;
            list->ResourceBarrier(1,&uav);
        };
        run(luma_,0,(qW+7)/8,(qH+7)/8,1);
        run(search_,0,kCoarseCount+1,1,2);
        run(solve_,0,1,1,1);
        run(search_,1,kRefineCount,1,2);
        run(solve_,1,1,1,1);
        run(mask_,0,(qW+7)/8,(qH+7)/8,1);
        run(update_,0,(bandTexels+2+7)/8,(kPictureH+2+7)/8,2);
        run(copy_,0,(qW+7)/8,(qH+7)/8,1);
        std::swap(b.Transition.StateBefore,b.Transition.StateAfter);
        list->ResourceBarrier(1,&b);
    }

    // After the main blit, on the bound render target, viewport and scissor.
    // dims are PresentBlit's 16 constants for this present.
    void compose(ID3D12GraphicsCommandList* list,const float* dims,DXGI_FORMAT targetFormat) {
        if(!ready_)return;
        if(targetFormat!=compositeFormat_){
            composite_={};compositeFormat_=targetFormat;
            if(!createComposite()){fail("composite pipeline");return;}
        }
        std::array<float,32> c{};
        std::copy(dims,dims+16,c.begin());
        c[16]=float(kCanvasW);c[17]=float(kCanvasH);c[18]=float(pictureW_);c[19]=float(kPictureH);
        c[20]=settings_.holdFrames;c[21]=settings_.fadeFrames;c[22]=settings_.strength;
        c[24]=settings_.dimNear;c[25]=settings_.dimFar;c[26]=settings_.blurNear;c[27]=settings_.blurFar;
        transitionShared(list,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        ID3D12DescriptorHeap* heaps[]={composite_.heap.Get()};list->SetDescriptorHeaps(1,heaps);
        list->SetGraphicsRootSignature(composite_.rootSig.Get());list->SetPipelineState(composite_.pso.Get());
        list->SetGraphicsRoot32BitConstants(0,32,c.data(),0);
        list->SetGraphicsRootDescriptorTable(1,composite_.heap->GetGPUDescriptorHandleForHeapStart());
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3,1,0,0);
        transitionShared(list,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

private:
    // Keep in step with shaders/EdgeFill.hlsli.
    static constexpr unsigned kPictureH=1080,kCanvasW=2048,kCanvasH=1536;
    static constexpr unsigned kCoarseCount=(2*40+1)*(2*10+1),kRefineCount=7*7;
    static constexpr unsigned kSrvSlots=2,kUavCount=11,kConstants=24;
    enum Texture:unsigned{CurHalf,PrevHalf,CurQuarter,PrevQuarter,Mask,Sad,State,CanvasL,CanvasR,StampL,StampR,TextureCount};

    static Settings loadSettings() {
        Settings s;
        if(const char* env=std::getenv("VEYRA_EDGE_FILL");env&&env[0]=='0')s.enabled=false;
        std::ifstream in(runtime::shaderDirectory()/"EdgeFill.ini");
        std::string line;
        while(std::getline(in,line)){
            const auto eq=line.find('=');
            if(eq==std::string::npos||line.empty()||line[0]=='#'||line[0]==';')continue;
            auto trim=[](std::string v){
                const auto a=v.find_first_not_of(" \t\r"),z=v.find_last_not_of(" \t\r");
                return a==std::string::npos?std::string():v.substr(a,z-a+1);
            };
            const std::string key=trim(line.substr(0,eq)),value=trim(line.substr(eq+1));
            const float v=float(std::atof(value.c_str()));
            if(key=="enabled")s.enabled=value!="0"&&value!="false";
            else if(key=="strength")s.strength=std::clamp(v,0.0f,1.0f);
            else if(key=="hold_frames")s.holdFrames=std::max(0.0f,v);
            else if(key=="fade_frames")s.fadeFrames=std::max(1.0f,v);
            else if(key=="dim_near")s.dimNear=std::clamp(v,0.0f,1.0f);
            else if(key=="dim_far")s.dimFar=std::clamp(v,0.0f,1.0f);
            else if(key=="blur_near")s.blurNear=std::clamp(v,0.0f,64.0f);
            else if(key=="blur_far")s.blurFar=std::clamp(v,0.0f,64.0f);
            else if(key=="band")s.band=std::clamp(v,0.05f,0.5f);
            else if(key=="mask_threshold")s.maskThreshold=std::clamp(v,0.0f,1.0f);
            else if(key=="exclude1"||key=="exclude2"){
                auto& rect=key=="exclude1"?s.exclude0:s.exclude1;
                std::array<float,4> r{};unsigned n=0;size_t at=0;
                while(n<4&&at<=value.size()){
                    const size_t comma=value.find(',',at);
                    r[n++]=std::clamp(float(std::atof(value.substr(at,comma-at).c_str())),0.0f,1.0f);
                    if(comma==std::string::npos)break;
                    at=comma+1;
                }
                if(n==4)rect=r;
            }
        }
        log::info("edge-fill",std::format("enabled={} strength={:.2f} hold={} fade={} dim={:.2f}/{:.2f} blur={:.1f}/{:.1f} band={:.2f}",
            s.enabled,s.strength,s.holdFrames,s.fadeFrames,s.dimNear,s.dimFar,s.blurNear,s.blurFar,s.band));
        return s;
    }
    void fail(const char* what) {
        log::warn("edge-fill",std::format("{} failed; edge history fill is off for this session",what));
        failed_=true;release();
    }
    bool createPasses() {
        const auto make=[&](pipeline::ComputePass& pass,const char* file){
            std::vector<uint8_t> cs;
            return pipeline::loadShaderBytes(file,cs)&&pass.create(device_,cs,1,1,kUavCount,kConstants,0,true);
        };
        if(!make(luma_,"EdgeLuma.dxil")||!make(search_,"EdgeSearch.dxil")||!make(solve_,"EdgeSolve.dxil")||
           !make(mask_,"EdgeMask.dxil")||!make(update_,"EdgeUpdate.dxil")||!make(copy_,"EdgeCopy.dxil"))return false;
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors=kSrvSlots+kUavCount;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if(FAILED(device_->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap_))))return false;
        increment_=device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return true;
    }
    static constexpr DXGI_FORMAT formatOf(unsigned t) {
        switch(t){
        case Sad:return DXGI_FORMAT_R32G32_FLOAT;
        case State:return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case CanvasL:case CanvasR:return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case StampL:case StampR:return DXGI_FORMAT_R32_UINT;
        default:return DXGI_FORMAT_R16_FLOAT;
        }
    }
    bool createResources() {
        const unsigned sizes[TextureCount][2]={
            {pictureW_/2,kPictureH/2},{pictureW_/2,kPictureH/2},{pictureW_/4,kPictureH/4},{pictureW_/4,kPictureH/4},
            {pictureW_/4,kPictureH/4},{kCoarseCount+1,2},{8,1},
            {kCanvasW,kCanvasH},{kCanvasW,kCanvasH},{kCanvasW,kCanvasH},{kCanvasW,kCanvasH}};
        const auto heap=heap_->GetCPUDescriptorHandleForHeapStart();
        for(unsigned t=0;t<TextureCount;++t){
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC td{};td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            td.Width=sizes[t][0];td.Height=sizes[t][1];td.DepthOrArraySize=1;td.MipLevels=1;
            td.Format=formatOf(t);td.SampleDesc.Count=1;td.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            textures_[t].Reset();
            // Committed resources start zeroed: stamp 0 = never written, frame 0.
            const HRESULT hr=device_->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&td,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&textures_[t]));
            if(FAILED(hr)){log::error("edge-fill",std::format("texture {} {}x{} hr=0x{:X}",t,sizes[t][0],sizes[t][1],unsigned(hr)));return false;}
            textures_[t]->SetName(L"Veyra edge fill");
            pipeline::makeUav(device_,textures_[t].Get(),formatOf(t),{heap.ptr+size_t(kSrvSlots+t)*increment_});
        }
        compositeFormat_=DXGI_FORMAT_UNKNOWN; // rebind the composite's views
        return true;
    }
    bool createComposite() {
        std::vector<uint8_t> vs,ps;
        if(!pipeline::loadShaderBytes("EdgeFillComposite_vs.dxil",vs)||!pipeline::loadShaderBytes("EdgeFillComposite_ps.dxil",ps)||
           !composite_.create(device_,vs,ps,5,compositeFormat_,32,5,true))return false;
        const auto heap=composite_.heap->GetCPUDescriptorHandleForHeapStart();
        const unsigned order[5]={CanvasL,CanvasR,StampL,StampR,State};
        for(unsigned i=0;i<5;++i)
            pipeline::makeSrv(device_,textures_[order[i]].Get(),formatOf(order[i]),{heap.ptr+size_t(i)*composite_.increment});
        return true;
    }
    void transitionShared(ID3D12GraphicsCommandList* list,D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to) {
        std::array<D3D12_RESOURCE_BARRIER,5> barriers{};
        const unsigned order[5]={CanvasL,CanvasR,StampL,StampR,State};
        for(unsigned i=0;i<5;++i){
            auto& b=barriers[i];b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition.pResource=textures_[order[i]].Get();
            b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;b.Transition.StateBefore=from;b.Transition.StateAfter=to;
        }
        list->ResourceBarrier(UINT(barriers.size()),barriers.data());
    }

    Settings settings_{};
    ID3D12Device* device_=nullptr;
    DXGI_FORMAT format_=DXGI_FORMAT_UNKNOWN,compositeFormat_=DXGI_FORMAT_UNKNOWN;
    unsigned pictureW_=0;
    bool ready_=false,failed_=false,resetPending_=true;
    pipeline::ComputePass luma_,search_,solve_,mask_,update_,copy_;
    pipeline::GraphicsPass composite_;
    pipeline::ComPtr<ID3D12DescriptorHeap> heap_;
    UINT increment_=0;
    std::array<pipeline::ComPtr<ID3D12Resource>,TextureCount> textures_{};
    std::array<uint32_t,kConstants> constants_{};
};

}
