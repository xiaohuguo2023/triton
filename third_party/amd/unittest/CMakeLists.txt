add_triton_ut(
  NAME AMDPerfModel
  SRCS PerfModelTest.cpp
  LIBS TritonAMDPerfModel LLVMSupport
)
