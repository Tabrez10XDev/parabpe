#pragma once

// macOS sysctl reads, shared by the hardware report (io.cpp), --merge-threads auto (trainer.cpp)
// and the trace's P/E core map (trace.cpp). Elsewhere these helpers don't exist.

#if defined(__APPLE__)
#include <sys/sysctl.h>

#include <string>

namespace parabpe::detail {

// An integer sysctl, or -1 when the name doesn't exist.
inline int sysctl_int(const char* name) {
  int value = 0;
  size_t size = sizeof(value);
  return sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : -1;
}

// A string sysctl, or `fallback` when the name doesn't exist.
inline std::string sysctl_string(const char* name, const std::string& fallback = "") {
  size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return fallback;
  std::string s(size, '\0');
  if (sysctlbyname(name, s.data(), &size, nullptr, 0) != 0) return fallback;
  s.resize(s.find('\0') == std::string::npos ? size : s.find('\0'));
  return s;
}

}  // namespace parabpe::detail
#endif
