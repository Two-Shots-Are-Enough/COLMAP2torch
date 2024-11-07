# COLMAP2torch

### format_changer.py
- COLMAP 형식의 디렉터리를 입력받아 realestate10k torch 형식으로 변환하는 코드.
- 추가로 JSON 파일도 생성되도록 설정함.

<br>

#### Issue (11-08 00:54)
- 이제 돌아감. 근데 결과가 노답임.
- 그래도 rotation matrix 자리에 rotation matrix가 있고, translation 자리에 translation이 있고, ... index.json도 맞고, evaluation_index_mipnerf.json(context index와 target index 전달)도 정상 작동함.
- c2w 에서 w2c로의 변환은 잘 된 것이 맞는 듯한데, 그니까 torch format changer에는 문제가 없는 듯한데.
- mipnerf.yaml을 막 짰는데 여기 설정이 엄밀히 필요한가? 예컨대 dtu는 yaml에서 near: 2.125, far: 4.525라고 넣어주고 있다..
  
