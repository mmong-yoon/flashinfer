# Paged KV-Cache 관련 코드 개요

이 문서는 FlashInfer 저장소에서 "paged" 구조의 KV-Cache를 다루는 주요 코드 파일들을 한눈에 파악하기 위해 작성되었습니다. 각 파일이 담당하는 역할과 서로의 연관성을 한국어로 정리합니다.

## 1. 핵심 자료구조 및 CUDA 커널

### `include/flashinfer/page.cuh`
- `paged_kv_t`, `paged_kv_mla_t` 구조체를 정의하여 페이지 단위 KV-Cache 레이아웃과 인덱스 관리 방식을 규정합니다.
- Decode/prefill 단계에서 새 토큰을 추가하기 위한 `AppendPagedKVCacheKernel`, `AppendPagedKVMlaCacheKernel` 등 CUDA 커널이 포함되어 있으며, 이를 호출하는 `AppendPagedKVCache`, `AppendPagedKVMlaCache` 함수도 제공합니다.

### `csrc/page.cu`
- 위 헤더에 선언된 커널을 PyTorch 연산으로 노출하는 구현 파일입니다. `append_paged_kv_cache`, `append_paged_mla_kv_cache`, `block_sparse_indices_to_vector_sparse_offsets` 등을 정의하여 Python 단에서 KV 페이지 관리 기능을 사용할 수 있게 합니다.

## 2. Prefill 단계의 paged KV 처리

### `include/flashinfer/attention/prefill.cuh`
- 배치 프리필(attention 계산과 KV 갱신)을 위한 핵심 CUDA 커널이 구현되어 있으며, `paged_kv_t`를 입력으로 받아 페이지 기반 KV-Cache에서 필요한 데이터를 읽고 씁니다.
- RoPE 적용, 마스킹, 병렬화 세부 전략 등이 함께 포함되어 있습니다.

### `include/flashinfer/attention/default_prefill_params.cuh`
- 프리필 커널 호출 시 사용하는 파라미터 구조체를 정의합니다. `paged_kv_t` 인스턴스를 보관하고 시퀀스 길이 계산 등을 제공하여 커널이 KV 정보를 손쉽게 참조하도록 돕습니다.

### `flashinfer/prefill.py`
- Python 레벨에서 paged KV-Cache 프리필/append 작업을 수행하기 위한 래퍼 클래스와 유틸리티 함수들을 제공합니다. `BatchPrefillWithPagedKVCacheWrapper`가 대표적이며, 내부적으로 위 C++/CUDA 커널을 호출합니다.

## 3. Decode 단계의 paged KV 처리

### `include/flashinfer/attention/decode.cuh`
- 여러 요청을 동시에 디코드할 때 사용되는 FlashAttention 기반 CUDA 커널을 정의합니다. 페이지 인덱스(`indptr`, `indices`)와 페이지 크기를 활용해 이전 토큰을 빠르게 로드하며, `paged_kv_t`로부터 시퀀스 길이와 실제 주소를 계산합니다.

### `include/flashinfer/attention/decode_mla_cute_sm80.cuh`
- SM80 아키텍처에서 MLA(Multi-Level Attention)와 paged 캐시를 결합한 특수 커널 버전입니다. 페이지 오프셋을 계산하여 압축된 KV(ckv)와 위치 임베딩(kpe)을 불러오는 로직이 포함됩니다.

### `include/flashinfer/attention/default_decode_params.cuh`
- 디코드 커널에서 사용할 파라미터 구조체를 정의합니다. `paged_kv_t` 또는 `paged_kv_mla_t`를 멤버로 가지고 있으며, 로프 스케일·인덱스 포인터 등 부가 정보를 저장합니다.

### `include/flashinfer/attention/scheduler.cuh`
- 배치 디코드 실행 전, 각 요청의 시퀀스 길이와 페이지 수를 분석해 적절한 그리드 크기 및 파티션 전략을 계산합니다. `DecodePlan` 함수는 paged KV 배열(`indptr`, `indices`, `last_page_len`)을 기반으로 필요한 임시 버퍼 크기와 블록 배치 정보를 산출합니다.

### `flashinfer/decode.py`
- 위 커널들을 Python에서 호출할 수 있도록 `get_batch_decode_jit_module` 등이 정의되어 있습니다. 계획 단계(`plan`)와 실행 단계(`run`)가 분리되어 있어 대규모 배치 디코드를 효율적으로 수행합니다.

### `include/flashinfer/attention/pod.cuh`
- Prefix-on-Demand(POD) 모드에서 프리필과 디코드를 혼합 실행할 때 사용하는 커널입니다. 내부에서 `decode_params.paged_kv`를 참조하여 페이지 기반 캐시와 연동합니다.

## 4. 기타 지원 코드

### `flashinfer/page.py`
- 페이지 기반 KV 관리용 Python API를 모아둔 모듈입니다. `append_paged_kv_cache`, `append_paged_mla_kv_cache` 등의 커스텀 연산을 등록하고, 배치 인덱스 계산과 시퀀스 길이 변환과 같은 보조 함수를 제공합니다.

### `flashinfer/utils.py`
- `_unpack_paged_kv_cache` 등 KV-Cache 레이아웃을 확인하고 텐서 형태로 분리하는 함수들이 포함되어 있어, Python 측 코드에서 paged 캐시 텐서를 손쉽게 다룰 수 있습니다.

### `csrc/flashinfer_ops.cu`
- 다양한 CUDA 커널을 PyTorch 모듈로 묶어 노출하는 파일로, paged 프리필/디코드 계획·실행 함수와 페이지 갱신 연산을 모두 등록합니다.

## 5. 동작 흐름 요약
1. **초기화 및 계획**: 사용자 코드는 `prefill.py` 혹은 `decode.py`의 래퍼를 통해 `plan` 함수를 호출합니다. 이 과정에서 `scheduler.cuh`의 로직이 실행되어 요청별 시퀀스 길이와 페이지 배치를 계산합니다.
2. **프리필/디코드 실행**: 계산된 `DecodePlanInfo` 또는 프리필 파라미터를 바탕으로 CUDA 커널(`prefill.cuh`, `decode.cuh` 등)이 호출되어, `paged_kv_t` 구조체에서 실제 페이지 주소를 가져와 연산을 수행합니다.
3. **KV 갱신**: 새 토큰이 생성되면 `page.py`의 `append_paged_kv_cache`(또는 MLA 버전)를 통해 CUDA 커널(`page.cuh`/`page.cu`)이 실행되어 KV-Cache에 페이지 단위로 값을 추가합니다.

이와 같이 여러 파일이 유기적으로 작동하여 paged 형태의 KV-Cache를 관리하고, 프리필·디코드·추가(append) 단계에서 효율적인 attention 연산을 가능하게 합니다.
