#include "command-program.h"
#include "graph-ir.h"
#include "qwen-program.h"
#include "reactive-plan.h"
#include "schedule.h"
#include "tool-utils.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using ggml::hrx::tool::write_file;

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::cerr << "usage: ggml-hrx-dump-program normalized-graph.json target corpus-manifest.json output-dir\n";
        return 2;
    }

    try {
        const std::filesystem::path graph_path = argv[1];
        const std::string target = argv[2];
        const std::filesystem::path corpus_path = argv[3];
        const std::filesystem::path output_directory = argv[4];

        const std::string graph_text = ggml::hrx::tool::read_file(graph_path);
        if (graph_text.empty()) throw std::runtime_error("cannot read " + graph_path.string());
        const ggml::hrx::Graph graph = ggml::hrx::deserialize_graph_json(graph_text);
        if (!graph.valid()) throw std::runtime_error(
            "invalid normalized graph: " + (graph.errors.empty() ? std::string("unknown error") : graph.errors.front()));

        const ggml::hrx::ProgramPlan plan = ggml::hrx::build_reactive_plan(graph, target);
        if (!plan.valid()) throw std::runtime_error(
            "cannot recover program: " + (plan.errors.empty() ? std::string("unknown error") : plan.errors.front()));

        std::vector<std::string> corpus_errors;
        const ggml::hrx::KernelCorpus corpus =
            ggml::hrx::load_kernel_corpus_manifest(corpus_path.string(), target, corpus_errors);
        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(plan, corpus);
        const ggml::hrx::VerificationResult verification =
            ggml::hrx::verify_command_program(plan, corpus, commands);

        std::filesystem::create_directories(output_directory);
        const ggml::hrx::QwenProgramProof proof = ggml::hrx::recover_owned_qwen3_moe_program(graph);
        const std::string readable_program = proof.recognized()
            ? ggml::hrx::qwen_program_signature(proof) : plan.semantic_witness;
        write_file(output_directory / "program.txt", readable_program);
        write_file(output_directory / "program-summary.txt", readable_program);
        write_file(output_directory / "semantic-witness.txt", plan.semantic_witness);
        write_file(output_directory / "program.json", ggml::hrx::serialize_schedule_json(plan.schedule));
        write_file(output_directory / "resources.txt", ggml::hrx::format_resource_program(plan.resources));
        write_file(output_directory / "kernels.txt", ggml::hrx::format_kernel_corpus(corpus));
        write_file(output_directory / "kernels.json", ggml::hrx::serialize_kernel_corpus_json(corpus));
        write_file(output_directory / "commands.txt", ggml::hrx::format_command_program(commands));
        write_file(output_directory / "commands.json", ggml::hrx::serialize_command_program_json(commands));
        write_file(output_directory / "commands.dot", ggml::hrx::command_program_dot(commands));

        std::vector<std::string> errors = corpus_errors;
        errors.insert(errors.end(), verification.errors.begin(), verification.errors.end());
        std::ostringstream status;
        status << "schema=ggml-hrx-plan-diagnostics-v1\n"
               << "workload=" << plan.schedule.workload << '\n'
               << "target=" << target << '\n'
               << "graph=" << graph.fingerprint << '\n'
               << "operations=" << graph.operations.size() << '\n'
               << "dispatches=" << ggml::hrx::schedule_dispatch_count(plan.schedule) << '\n'
               << "commands=" << commands.commands.size() << '\n'
               << "valid=" << (errors.empty() ? "true" : "false") << '\n'
               << ggml::hrx::format_verification_summary(errors);
        write_file(output_directory / "status.txt", status.str());
        write_file(output_directory / "verification-errors.txt", ggml::hrx::format_verification_errors(errors));

        std::cout << "dumped " << plan.schedule.workload << " graph=" << graph.fingerprint
                  << " dispatches=" << ggml::hrx::schedule_dispatch_count(plan.schedule)
                  << " commands=" << commands.commands.size()
                  << " valid=" << (corpus_errors.empty() && verification.valid() ? "true" : "false")
                  << " to " << output_directory << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ggml-hrx-dump-program: " << error.what() << '\n';
        return 1;
    }
}
