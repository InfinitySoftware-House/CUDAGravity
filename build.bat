@echo off
REM Build the 3D CUDA Barnes-Hut N-body simulation on Windows.
REM Requires: CUDA Toolkit, CMake, Git, and MSVC Build Tools.

setlocal

REM --- locate and load the MSVC x64 environment (needed by nvcc) ---
set "VSDEV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VSDEV%" set "VSDEV=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VSDEV%" (
  echo Could not find vcvars64.bat - edit build.bat to point at your VS install.
  exit /b 1
)
call "%VSDEV%"

cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

cmake --build build
if errorlevel 1 exit /b 1

echo.
echo Done.  Run:  build\gravity.exe
echo        Bench: build\gravity.exe --bench --n=100000
echo        Verify:build\gravity.exe --verify
