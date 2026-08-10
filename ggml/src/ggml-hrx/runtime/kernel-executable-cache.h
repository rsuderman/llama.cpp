#pragma once

#include "dispatch/dispatch.h"
#include "hrx_runtime.h"
#include "kernel-corpus/kernel-corpus.h"
#include "loom-jit.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml::hrx {

struct KernelExecutableArtifact {
    ~KernelExecutableArtifact();

    hrx_executable_t                executable     = nullptr;
    uint32_t                        export_ordinal = 0;
    hrx_executable_export_info_t    export_info    = {};
    ggml_hrx_loom_jit_launch_config launch;
};

struct KernelExecutablePrepareContext {
    hrx_device_t                device = nullptr;
    const char *                target = nullptr;
    ggml_hrx_loom_jit_amdgpu ** jit    = nullptr;
};

class KernelExecutableCache {
  public:
    std::shared_ptr<KernelExecutableArtifact> prepare(const KernelExecutablePrepareContext & context,
                                                      const KernelDefinition &               definition,
                                                      const Dispatch &                       dispatch,
                                                      std::vector<uint8_t> &                 constants);

    void clear();

  private:
    std::mutex                                                                 mutex_;
    std::unordered_map<std::string, std::shared_ptr<KernelExecutableArtifact>> cache_;
};

}  // namespace ggml::hrx
