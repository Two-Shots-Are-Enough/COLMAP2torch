# COLMAP2torch

### format_changer.py
- COLMAP 형식의 디렉터리를 입력받아 realestate10k torch 형식으로 변환하는 코드.
- 추가로 JSON 파일도 생성되도록 설정함.

<br>

#### Issue (11-08 00:54)
- 이제 돌아감. 근데 결과가 노답임.
- 그래도 rotation matrix 자리에 rotation matrix가 있고, translation 자리에 translation이 있고, ... index.json도 맞고, evaluation_index_mipnerf.json(context index와 target index 전달)도 정상 작동함.
- 수치적인 문제(coordinate or normalization or ...)가 있는 듯하다.
