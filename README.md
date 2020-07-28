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
- '코드' 다운로드 ([link](https://github.com/jjun0127/melon_autoencoder/archive/master.zip))
- '데이터' 다운로드 ([link](https://arena.kakao.com/c/7/data))
  - `train.json`, `val.json`, `test.json`, `genre_gn_all.json`, `song_meta.json` 파일 다운
  - arena_mel_n.tar 파일들은 다운 x
- '모델' 다운로드
  - [test용 모델](https://drive.google.com/file/d/1tAXY8iMpUt-Uft8RWZgi2Mub69-TEaUi/view?usp=sharing)
- '중간 파일' 다운로드
  - [test용 중간 파일](https://drive.google.com/file/d/1Lr-IxR3kJzhFXYkh03H8aURWwiJkxPPp/view?usp=sharing)
  - 빠른 추론을 위한 것으로 모델 크기 제한 1GB에 포함되지 않습니다.
~~~
.
├── model
│   └── # 다운로드한 '모델' 파일들을 넣어주세요.
├── res
│   └── # 다운로드한 '데이터' 파일들을 넣어주세요.
├── results
├── scores
│   └── # 다운로드한 '중간파일' 넣어주세요.
├── arena_data
│   ├── answers
│   ├── orig
│   └── questions
└── # 다운로드한 '코드' 파일들을 넣어주세요. 
~~~

## 5. 코드 실행 방법 & 추천 결과 재현 방법
### 1) 공개된 test case에 대해 inference할 경우
**<STEP 1>** `$> python train.py` 실행
  - **4번에서 "test용 모델"을 다운 받으시면 STEP 1은 skip하시면 됩니다.**
  - test.json에 대한 추천결과 재현을 위해 default 값이 설정되어 있습니다.
  - 입력 인자 
    - `dimensions`: size of hidden layer dimension
    - `epochs`: number of total epochs
    - `batch_size`: batch size
    - `learning_rate`: learning rate
    - `dropout`: dropout
    - `num_workers`: num workers
    - `freq_thr`: frequency threshold (주어진 값 이상의 빈도를 갖는 song들만 사용)
    - `mode`: local_val: 0 / val: 1 / test: 2 (default=2)
  - 출력 파일
    - `freq_song2id`, `id2freq_song`,: Song One-hot Vector 생성을 위한 파일
    - `tag2id`, `id2tag`: Tag One-hot Vector 생성을 위한 파일
    - `autoencoder_{}_{}_{}_{}_{}_{mode 명}.pkl`: 학습된 AutoEncoder 모델 파일
    - `tokenizer_{}_{}_{mode 명}.model`: 학습된 Tokenizer 모델 파일
    - `w2v_{}_{}_{mode 명}.model`: 학습된 Word2Vec 모델 파일
    
**<STEP 2>** `$> python inference.py` 실행
  - **4번에서 "test용 중간 파일"을 다운 받으시면 빠른 추론이 가능합니다.**
  - test.json에 대한 추천결과 재현을 위해 default 값이 설정되어 있습니다.
  - 입력 인자 
    - `mode`: local_val: 0 / val: 1 / test: 2 (default=2)
  - 출력 파일
    - `test_scores_bias_cos.npy`: 학습된 AutoEncoder 기반으로 계산한 플레이리스트 사이의 Cosine Similarity
    - `test_scores_bias_cos_gnr.npy`: 학습된 AutoEncoder 기반에 장르 정보를 추가하여 계산한 플레이리스트 사이의 Cosine Similarity
    - `test_scores_title_cos_24000.npy`: 학습된 Tokenizer와 Word2Vec 기반으로 계산한 플레이리스트 사이의 Cosine Similarity
    - `results_{종료 시각}_{mode 명}.json`: 최종 추천 결과 파일
    
 ### 2) 새로운 test case에 대해 inference할 경우
**<STEP 1>** test용으로 1)에서 학습된 모델을 사용하므로 train.py를 실행할 필요가 없습니다. 
  
**<STEP 2>** 새롭게 inference할 플레이리스트를 res 폴더 안에 파일 명을 `test.json`으로 하여 넣어주세요.  

**<STEP 3>** scores 폴더 안에 기존 scores 파일들은 삭제해주세요. (새로운 test case에 대한 score를 새롭게 구하기 위해) 
  
**<STEP 4>** `$> python inference.py` 실행
