#pragma once

#include "matcher.h"
#include "schedule.h"

#include <vector>

namespace ggml::hrx {

struct FusionRule {
    MatchAutomaton automaton;
    KernelSpecialization kernel;
    int priority = 0;
};

struct SelectedRegion {
    size_t rule = 0;
    OperationId root = kInvalidId;
    std::vector<OperationId> operations;
};

struct Selection {
    std::vector<SelectedRegion> regions;
    std::vector<OperationId> uncovered_operations;
};

Selection select_regions(const Graph & graph, const std::vector<FusionRule> & rules);
Schedule materialize_schedule(const Graph & graph, const std::vector<FusionRule> & rules, const Selection & selection);
Schedule materialize_schedule_with_cpu_fallback(const Graph & graph, const std::vector<FusionRule> & rules, const Selection & selection);

} // namespace ggml::hrx
