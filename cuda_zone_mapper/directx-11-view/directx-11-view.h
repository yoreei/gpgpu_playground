#pragma once

#include "resource.h"

// Inside Dx11View.h
#ifdef DIRECTX11VIEW_EXPORTS
#define DIRECTX11VIEW_API __declspec(dllexport)
#else
#define DIRECTX11VIEW_API __declspec(dllimport)
#endif

extern "C" DIRECTX11VIEW_API int show_window();
