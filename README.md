# COLMAP2torch

### format_changer.py
- COLMAP 형식의 디렉터리를 입력받아 realestate10k torch 형식으로 변환하는 코드.
- 추가로 JSON 파일도 생성되도록 설정함.

<br>

#### Issue (11-07 23:28)
- 카메라 행렬이 올바르지 않아 `MVSplat/src/dataset/dataset_re10k.py`에서 에러가 발생하고 있음. => 주석 단 `dataset_re10k.py` 업로드.
- 현재 `dataset_re10k.py`에 print로 디버깅 중이며, 코드에 적힌 대로 다음 두 경우의 **rotation matrix 및 그 determinant 출력값**이 다름:
  - `re10k.torch` (현재 MVSplat에서 inference가 잘 작동하는 데이터) 사용 시 출력값
  - `mipnerf.torch` (`format_changer.py`로 만든 torch 파일) 사용 시 출력값

- 여기서 mipnerf의 경우 rotation matrix의 determinant가 1이 되지 않고 있는 문제가, `src/misc/sh_rotation.py`의 `rotate_sh` 함수에 문제를 일으키는 것으로 보임.
