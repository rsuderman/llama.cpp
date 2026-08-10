#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

class ErrorLog {
  public:
    bool success() const { return messages_.empty(); }

    bool empty() const { return messages_.empty(); }

    size_t size() const { return messages_.size(); }

    const std::string & front() const {
        static const std::string empty_message;
        return messages_.empty() ? empty_message : messages_.front();
    }

    const std::vector<std::string> & messages() const { return messages_; }

    void append(const ErrorLog & other) {
        messages_.insert(messages_.end(), other.messages_.begin(), other.messages_.end());
    }

    void log(const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_va(format, args);
        va_end(args);
    }

  private:
    void push(std::string message) { messages_.push_back(std::move(message)); }

    void log_va(const char * format, va_list args) {
        if (format == nullptr) {
            push("failed to format error message");
            return;
        }

        char    stack[256];
        va_list args_copy;
        va_copy(args_copy, args);
        const int written = std::vsnprintf(stack, sizeof(stack), format, args_copy);
        va_end(args_copy);
        if (written < 0) {
            push("failed to format error message");
            return;
        }
        if (static_cast<size_t>(written) < sizeof(stack)) {
            push(std::string(stack, static_cast<size_t>(written)));
            return;
        }

        std::vector<char> buffer(static_cast<size_t>(written) + 1);
        const int         rewritten = std::vsnprintf(buffer.data(), buffer.size(), format, args);
        if (rewritten < 0) {
            push("failed to format error message");
            return;
        }
        push(std::string(buffer.data(), static_cast<size_t>(rewritten)));
    }

    std::vector<std::string> messages_;
};

}  // namespace ggml::hrx
