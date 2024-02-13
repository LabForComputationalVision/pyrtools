#pragma once

// this macro controls symbol visibility in the shared library
#if defined _WIN32 || defined __MINGW32__
#define PYRTOOLS_EXPORT __declspec(dllexport)
#else
#define PYRTOOLS_EXPORT
#endif
