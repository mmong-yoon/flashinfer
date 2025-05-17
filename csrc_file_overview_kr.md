# csrc 디렉터리 파일 개요

이 문서는 `csrc` 폴더에 존재하는 모든 파일의 기능을 간략히 정리한 것입니다. 파일 이름을 기준으로 대략적인 역할을 파악하여 기술하였으며, 각 파일 간의 상관관계도 함께 서술합니다.

## 파일별 설명
- **activation.cu**: SiLU, GeLU 등 활성화 함수를 CUDA 커널로 구현합니다.
- **aot_extension_utils.h**: AOT(사전 컴파일) 확장을 위한 디스패치 매크로와 헬퍼 함수들을 정의합니다.
- **batch_decode.cu**: 배치 디코드(Paged KV 캐시 사용) 연산의 계획과 실행을 담당합니다.
- **batch_decode_config.inc**: 배치 디코드 연산을 위한 기본 설정을 포함한 인클루드 파일입니다.
- **batch_decode_customize_config.jinja**: 배치 디코드 설정을 사용자 정의하여 생성하기 위한 Jinja 템플릿입니다.
- **batch_decode_jit_pybind.cu**: 배치 디코드 연산을 JIT 방식으로 파이썬 모듈에 바인딩합니다.
- **batch_decode_kernel_inst.jinja**: 배치 디코드 커널 인스턴스 생성을 위한 Jinja 템플릿입니다.
- **batch_decode_mla_config.jinja**: MLA(Multi-Level Attention) 버전 배치 디코드 설정 템플릿입니다.
- **batch_decode_mla_cute_sm80.cu**: SM80 아키텍처에서 MLA 디코드를 지원하는 전용 CUDA 구현입니다.
- **batch_decode_mla_plan.cu**: MLA 디코드 실행 전 계획(Plan) 수립을 담당합니다.
- **batch_decode_mla_pybind.cu**: MLA 디코드 관련 함수를 파이썬에 바인딩합니다.
- **batch_decode_mla_run.cu**: MLA 디코드를 실제 실행하는 코드입니다.
- **batch_mla_config.jinja**, **batch_mla_plan.cu**, **batch_mla_pybind.cu**, **batch_mla_run.cu**: 배치 MLA 연산을 위한 설정·계획·바인딩·실행 코드입니다.
- **batch_mla_sm90_plan.cu**, **batch_mla_sm90_pybind.cu**, **batch_mla_sm90_run.cu**: SM90 아키텍처 최적화 버전의 MLA 디코드 관련 파일들입니다.
- **batch_prefill.cu**: 배치 프리필(Prefill) 연산의 계획과 실행을 구현합니다.
- **batch_prefill_config.inc**: 배치 프리필 기본 설정을 모아 둔 인클루드 파일입니다.
- **batch_prefill_customize_config.jinja**: 배치 프리필 설정을 사용자 정의하기 위한 템플릿입니다.
- **batch_prefill_fp8_paged_sm90_kernel_inst.jinja**, **batch_prefill_fp8_ragged_sm90_kernel_inst.jinja**: FP8 형식의 프리필 커널을 SM90에서 생성하기 위한 Jinja 템플릿입니다.
- **batch_prefill_fp8_sm90.cu**: SM90 전용 FP8 프리필 커널 구현입니다.
- **batch_prefill_jit_pybind.cu**: 배치 프리필 연산을 파이썬과 연동하기 위한 JIT 바인딩 코드입니다.
- **batch_prefill_paged_kernel_inst.jinja**, **batch_prefill_paged_sm90_kernel_inst.jinja**, **batch_prefill_ragged_kernel_inst.jinja**, **batch_prefill_ragged_sm90_kernel_inst.jinja**: 프리필 커널 인스턴스 생성을 위한 Jinja 템플릿 모음입니다.
- **batch_prefill_sm90.cu**, **batch_prefill_sm90_config.inc**, **batch_prefill_sm90_customize_config.jinja**, **batch_prefill_sm90_jit_pybind.cu**: SM90 환경에서의 프리필 연산 구현과 설정, 파이썬 바인딩을 담당합니다.
- **bmm_fp8.cu**: FP8 데이터 타입을 이용한 batched matrix multiplication(BMM) 연산입니다.
- **cascade.cu**: 여러 self-attention 상태를 병합하는 기능(캐스케이드)을 제공합니다.
- **custom_all_reduce.cu**: 통신 성능 향상을 위한 맞춤형 AllReduce 연산 구현입니다.
- **cutlass_mla.cu**: CUTLASS 라이브러리를 활용한 MLA 연산 구현입니다.
- **flashinfer_cascade_ops.cu**: 캐스케이드 관련 PyTorch 연산을 바인딩합니다.
- **flashinfer_comm_ops.cu**: 통신(collective) 연산과 관련된 파이썬 바인딩 코드입니다.
- **flashinfer_gemm_ops.cu**, **flashinfer_gemm_sm90_ops.cu**: GEMM 연산의 일반 버전과 SM90 최적화 버전을 파이썬에 바인딩합니다.
- **flashinfer_mla_ops.cu**: MLA 연산들을 파이썬에 노출하는 파일입니다.
- **flashinfer_norm_ops.cu**: RMSNorm 등 정규화 관련 커널 바인딩을 제공하는 파일입니다.
- **flashinfer_ops.cu**, **flashinfer_ops_sm90.cu**: FlashInfer의 주요 연산을 통합하여 파이썬 모듈로 노출합니다. SM90 전용 파일도 포함됩니다.
- **flashinfer_page_ops.cu**: 페이지 단위의 KV 캐시 관리 등 페이지 연산을 바인딩합니다.
- **flashinfer_quantization_ops.cu**: 양자화 연산 관련 파이썬 바인딩입니다.
- **flashinfer_rope_ops.cu**: RoPE(회전 위치 인코딩) 관련 연산들을 바인딩합니다.
- **flashinfer_sampling_ops.cu**: 확률/로그잇 기반 샘플링 및 탑-k, 탑-p 연산들을 파이썬에 제공합니다.
- **fmha_cutlass_sm100.cu**, **fmha_cutlass_sm100_pybind.cu**: SM100 아키텍처용 Cutlass 기반 어텐션 커널과 그 파이썬 바인딩입니다.
- **gemm_groupwise_sm100.cu**, **gemm_sm100_pybind.cu**: 그룹별 GEMM 연산의 SM100 버전 및 바인딩 코드입니다.
- **group_gemm.cu**, **group_gemm_*_sm90.cu**, **group_gemm_groupwise_sm100.cu**, **group_gemm_sm100_pybind.cu**, **group_gemm_sm90.cu**: 다양한 데이터 타입과 아키텍처(SM90, SM100)에 대응하는 그룹 GEMM 커널 구현입니다.
- **norm.cu**, **renorm.cu**: RMSNorm, 재정규화 등 정규화 연산 커널을 구현합니다.
- **page.cu**: KV 캐시의 페이지 단위 조작을 수행하는 커널입니다.
- **pod.cu**, **pod_config.inc**, **pod_customize_config.jinja**, **pod_jit_pybind.cu**, **pod_kernel_inst.jinja**: POD(Prefix-on-Demand) 형태의 어텐션 연산 구현과 설정, 파이썬 바인딩을 제공합니다.
- **pytorch_conversion_utils.h**: std::vector와 PyTorch Tensor 간 형 변환 유틸리티 함수들입니다.
- **pytorch_extension_utils.h**: PyTorch C++ 확장 모듈을 간단히 작성할 수 있는 헬퍼 매크로들입니다.
- **quantization.cu**: 비트 패킹 등 GPU 기반 양자화 연산을 구현합니다.
- **rope.cu**: RoPE 적용을 위한 커널과 보조 함수를 정의합니다.
- **runtime_utils.h**: 런타임 환경에서 사용되는 여러 유틸리티가 들어 있습니다.
- **sampling.cu**: 샘플링 커널의 내부 구현부입니다.
- **single_decode.cu**: 단일 요청에 대한 디코드(append) 연산을 수행합니다.
- **single_decode_config.inc**, **single_decode_customize_config.jinja**, **single_decode_jit_pybind.cu**, **single_decode_kernel_inst.jinja**: 단일 디코드 연산의 설정, 템플릿, 파이썬 바인딩을 제공합니다.
- **single_prefill.cu**: 단일 요청 프리필 연산을 수행합니다.
- **single_prefill_config.inc**, **single_prefill_customize_config.jinja**, **single_prefill_fp8_sm90.cu**, **single_prefill_fp8_sm90_kernel_inst.jinja**, **single_prefill_jit_pybind.cu**, **single_prefill_kernel_inst.jinja**, **single_prefill_sm90.cu**, **single_prefill_sm90_config.inc**, **single_prefill_sm90_customize_config.jinja**, **single_prefill_sm90_jit_pybind.cu**, **single_prefill_sm90_kernel_inst.jinja**: 단일 프리필 연산 관련 설정과 SM90 최적화 버전, 파이썬 바인딩 파일들입니다.

## 파일 간 상관관계
- 여러 `*_config.inc`와 `*_customize_config.jinja` 파일은 커널 생성을 위한 설정을 정의하며, `*_kernel_inst.jinja`에서 실제 커널 인스턴스 코드를 생성합니다.
- `*_pybind.cu` 혹은 `*_jit_pybind.cu` 파일들은 위에서 구현된 CUDA 커널을 PyTorch에서 호출할 수 있도록 바인딩합니다.
- `*_plan.cu`와 `*_run.cu` 형식의 파일들은 대규모 배치 작업에서 사전 계획(Plan) 단계와 실행(Run) 단계를 나누어 처리하기 위한 구조를 가집니다.
- `flashinfer_ops.cu`는 다양한 개별 연산들을 한데 모아 단일 PyTorch 모듈로 노출하며, 하위의 `flashinfer_*_ops.cu` 파일들은 세부 기능별 바인딩을 담당합니다.
- SM80, SM90 등 아키텍처 이름이 붙은 파일들은 특정 GPU 아키텍처에서 최적화된 버전을 제공하여 성능을 높입니다.
- `runtime_utils.h`, `pytorch_extension_utils.h`, `pytorch_conversion_utils.h` 등은 다른 파일들이 공통적으로 참조하는 헬퍼 및 유틸리티 역할을 합니다.

해당 파일들은 전체적으로 대규모 언어 모델(LLM) 추론을 위한 다양한 CUDA 커널과 PyTorch 바인딩을 제공하며, 프리필·디코드·샘플링·정규화·양자화 등 LLM 서빙 과정에서 필요한 주요 기능을 담당합니다.
