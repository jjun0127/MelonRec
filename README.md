### kakao arena 3rd Competition (카카오 아레나 3차 대회)
# Melon Playlist Continuation
- **팀명**: 멜론이 체질 ([공개 리더보드](https://arena.kakao.com/c/7/leaderboard) 5위)
- **팀원**
  - jun.94
  - wonzzang
  - jjun0127
  - datartist
  
## 1. 대회 개요
- **목표**: 플레이리스트에 수록된 곡과 태그의 절반 또는 전부가 숨겨져 있을 때, 주어지지 않은 곡들과 태그를 예측하는 것
- **데이터**
  - 플레이리스트 메타데이터 (제목, 곡, 태그, 좋아요 수, 업데이트 시점)
  - 곡 메타데이터 (곡 제목, 앨범 제목, 아티스트명, 장르, 발매일)
  - 곡에 대한 Mel-spectrogram
- **참여 팀 수**: 786팀
  
## 2. 모델 설명
**<모델 개요>**
![model](https://user-images.githubusercontent.com/50820635/88534733-a9e58900-d043-11ea-821b-1166c64e2b42.png)

**<STEP 1>** 플레이리스트간 Similarity 계산  
- AutoEncoder로 Embedding
  - Input: Song One-hot Vector, Tag One-hot Vector 
  - Ouput: Playlist Embedding Vector
- Word2Vec으로 Embedding
  - Input: Sentence (Title, Tag, Genre, Date)
  - Ouput: Playlist Embedding Vector
- Embedding Vector간 Similarity 계산
  - Cosine Similarity
  - Euclidean Distance
  - Pearson Correlation Coefficient

**<STEP 2>** k개의 비슷한 플레이리스트로부터 Song Score, Tag Score 계산  
- k Nearest Neighbor의 Song과 Tag 활용
- Similarity가 높을수록 가중치
- Frequent Song, Same Artist 등의 정보로 Score 보정

**<STEP 3>** Song 추천, Tag 추천  
- 이미 담겨있는 중복 Song, Tag 제외
- 플레이리스트 업데이트 시점 이후의 Song 제외
- Song 100개, Tag 10개가 채워지지 않았다면 Popular로 채우기  

## 3. 개발 환경 및 라이브러리
### 1) 개발 환경
- OS: Windows 10
- CPU: i9-7900X
- GPU: GTX1080Ti
- RAM: 64GB
### 2) 라이브러리
- numpy 1.16.2
- pandas 0.24.2
- matplotlib 3.0.3
- tqdm 4.31.1
- gensim 3.8.3
- sentencepiece 0.1.91
- sklearn 0.20.3
- pytorch 1.5.1

## 4. 폴더 및 파일
- 코드 다운로드 ([link](https://github.com/jjun0127/melon_autoencoder/archive/master.zip))
- 데이터 다운로드 ([link](https://arena.kakao.com/c/7/data))
- 모델 다운로드
  - [test용 모델]()
- 중간 파일 다운로드
  - [test용 중간 파일]()
~~~
.
├── model
├── res
├── results
├── scores
└── arena_data
    ├── answers
    ├── orig
    └── questions
~~~

## 5. 추천 결과 재현 방법
- train.py 실행
 - 개발 환경이 갖추어진 상태에서 train.py를 실행합니다.
 - 하이퍼 파라미터들은 제출 시 사용한 값이 default 로 설정 되어있습니다. 아래는 모델 튜닝에 사용한 하이퍼 파라미터들 입니다.
  - dimensions: size of hidden layer dimension
  - epochs: number of total epochs
  - batch_size: batch size
  - learning_rate: learning rate
  - dropout: dropout
  - num_workers: num workers
  - freq_thr: frequency threshold, 주어진 값 이상의 빈도를 갖는 song들만 사용합니다. (tag는 적용 x)
  - mode: local_val: 0, val: 1, test: 2, test data 결과를 재현하기 위해서는 mode를 2로 하면 됩니다. (default=2)
