import pandas as pd
import tensorflow as tf
import enum
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
from configs import DEFINES


PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]

"""
데이터를 읽는 파트

load_data(): 데이터를 읽고 트레이닝 셋과 테스트 셋으로 분리
train_q: 트레이닝용 질문 데이터
train_a: 트레이닝용 대답 데이터
test_q: 테스트용 질문 데이터
test_a: 테스트용 대답 데이터
output: train_q, train_a, test_q, test_a
"""

def load_data():
    # 판다스를 통해서 데이터를 불러온다.
    data_df = pd.read_csv(DEFINES.data_path, header=0)
    # 질문과 답변 열을 가져와 question과 answer에 넣는다.
    question, answer = list(data_df['Q']), list(data_df['A'])
    # 학습 셋과 테스트 셋을 나눈다.
    train_q, test_q, train_a, test_a = train_test_split(question, answer, test_size=0.33,
                                                                        random_state=42)
    # 그 값을 리턴한다.
    return train_q, train_a, test_q, test_a


"""
문장 데이터 전 처리 파트 1

prepro_noise_canceling(data): 
텍스트 데이터에 정규화를 사용하여 ([~.,!?\"':;)(]) 제거
output: ([~.,!?\"':;)(]) 제거된 텍스트 데이터
"""

def prepro_noise_canceling(data):

    CHANGE_FILTER = re.compile("([~.,!?\"':;)(])")
    #  리스트를 생성합니다.
    result_data = list()

    for seq in data:
        
        seq = re.sub(CHANGE_FILTER, "", seq)
        result_data.append(seq)

    return result_data


"""
문장 데이터 전 처리 파트 2
tokenizing_data(data): 텍스트 데이터들 토크나이징
input:
data: 텍스트 데이터 행렬

데이터 노이즈 처리 함수 처리 ->
띄어쓰기 단위로 나누기 ->
띄어진 단어들 벡터 형성
output: [token1, token2, ...]
"""

def tokenizing_data(data):
    # 토크나이징 해서 담을 배열 생성
    doc = []
    data = prepro_noise_canceling(data)
    for sentence in data:
        words = []
        for word in sentence.split(): 
            words.append(word)
        doc.append(words)
    return doc


"""
enc_processing(value, dictionary): 인코더용 입력값 생성 함수
텍스트 데이터 -> 인덱스 벡터화 및 길이
input:
value: 텍스트 문장들 데이터
dictionary: 값이 인덱스인 단어 사전

value 값 노이즈 캔슬링 ->
문장 단위로 나누기 ->
문장을 토큰 단위로 나누기 ->
dictionary를 활용하여 토큰 인덱스화 ->
dictionary 없는 토큰의 경우 unk값으로 대체 ->
기준 문장 길이 보다 크게 된다면 뒤의 토큰 자르기 ->
기준 문장 길이에 맞게 남은 공간에 padding ->
문장 인덱스와 문장 길이 계산
output: 넘파이 문장 인덱스 벡터, 문장 길이 
"""

from konlpy.tag import Okt

def prepro_rewrite_konlpy(data):
    konlpy_analyzer = Okt()
    result_data = list()
    for seq in data:
        konlpy_seq = " ".join(konlpy_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(konlpy_seq)

    return result_data


def enc_processing(value, dictionary):
    
    # 인덱스 정보를 저장할 배열 초기화
    seq_input_index = []
    # 문장의 길이를 저장할 배열 초기화
    seq_len = []
    # 노이즈 캔슬
    
    for seq in value:
        
        # 하나의 seq에 index를 저장할 배열 초기화
        seq_index =[]
        
        for word in seq:
            if dictionary.get(word) is not None:
                # seq_index에 dictionary 안의 인덱스를 extend 한다
                seq_index.extend([dictionary[word]])
            else:
                # dictionary에 존재 하지 않는 다면 UNK 값을 extend 한다 
                seq_index.extend([dictionary[UNK]])
                
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 제거
        if len(seq_index) > DEFINES.max_sequence_length:
            seq_index = seq_index[:DEFINES.max_sequence_length]
            
        # seq의 길이를 저장
        seq_len.append(len(seq_index))
        
        # DEFINES.max_sequence_length 길이보다 작은 경우 PAD 값을 추가 (padding)
        seq_index += (DEFINES.max_sequence_length - len(seq_index)) * [dictionary[PAD]]
        
        # 인덱스화 되어 있는 값은 seq_input_index에 추가
        seq_input_index.append(seq_index)
        
    return np.asarray(seq_input_index)

"""
dec_input_processing(value, dictionary): 디코더용 입력값 생성 함수
텍스트 데이터 -> 인덱스 벡터화 및 길이
input:
value: 텍스트 문장들 데이터
dictionary: 값이 인덱스인 단어 사전

value 값 노이즈 캔슬링 ->
문장 단위로 나누기 ->
문장을 토큰 단위로 나누기 ->
dictionary를 활용하여 토큰 인덱스화 ->
인덱스 앞에 STD 추가 ->
기준 문장 길이 보다 크게 된다면 뒤의 토큰 자르기 ->
기준 문장 길이에 맞게 남은 공간에 padding ->
문장 인덱스와 문장 길이 계산
output: 넘파이 문장 인덱스 벡터, 문장 길이 
"""

def dec_input_processing(value, dictionary):
    
    # 인덱스 정보를 저장할 배열 초기화
    seq_input_index = []
    # 문장의 길이를 저장할 배열 초기화
    seq_len = []
    
    
    for seq in value:
        # 하나의 seq에 index를 저장할 배열 초기화
        seq_index =[]
        
        for word in seq:
            
            
            if dictionary.get(word) is not None:
                # seq_index에 dictionary 안의 인덱스를 extend 한다
                seq_index.extend([dictionary[word]])
            else:
                # dictionary에 존재 하지 않는 다면 seq_index에 UNK 값을 extend 한다 
                seq_index.extend([dictionary[UNK]])

            # 디코딩 입력의 처음에는 START가 와야 하므로 STD 값 추가
            seq_index = [dictionary[STD]] + seq_index
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 제거
        if len(seq_index) > DEFINES.max_sequence_length:
            seq_index = seq_index[:DEFINES.max_sequence_length]
            
        # seq의 길이를 저장
        seq_len.append(seq_index)
        
        # DEFINES.max_sequence_length 길이보다 작은 경우 PAD 값을 추가 (padding)
        seq_index += (DEFINES.max_sequence_length - len(seq_index)) * [dictionary[PAD]]
        
        # 인덱스화 되어 있는 값은 seq_input_index에 추가
        seq_input_index.append(seq_index)
    
    return np.asarray(seq_input_index)

"""
dec_target_processing(value, dictionary): 디코더용 입력값 생성 함수
텍스트 데이터 -> 인덱스 벡터화 및 길이
input:
value: 텍스트 문장들 데이터
dictionary: 값이 인덱스인 단어 사전

value 값 노이즈 캔슬링 ->
문장 단위로 나누기 ->
문장을 토큰 단위로 나누기 ->
dictionary를 활용하여 토큰 인덱스화 ->
기준 문장 길이 보다 크게 된다면 뒤의 토큰 자르기 ->
인덱스 뒤에 END 추가 ->
기준 문장 길이에 맞게 남은 공간에 padding ->
문장 인덱스와 문장 길이 계산
output: 넘파이 문장 인덱스 벡터, 문장 길이 
"""

def dec_target_processing(value, dictionary):
    
    # 인덱스 정보를 저장할 배열 초기화
    seq_input_index = []
    # 문장의 길이를 저장할 배열 초기화
    seq_len = []
   
    
    for seq in value:
        
        # 하나의 seq에 index를 저장할 배열 초기화
        seq_index =[]
        
        for word in seq:
            # 잘려진 단어들이 딕셔너리에 존재 하는지 보고
            # 그 값을 가져와 sequence_index에 추가한다.
            if dictionary.get(word) is not None:
                seq_index.extend([dictionary[word]])
            # 잘려진 단어가 딕셔너리에 존재 하지 않는
            # 경우 이므로 UNK(2)를 넣어 준다.
            else:
                seq_index.extend([dictionary[UNK]])
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 제거
        # END 토큰을 추가 (DEFINES.max_sequence_length 길이를 맞춰서 추가)
        
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        if len(seq_index) >= DEFINES.max_sequence_length:
            seq_index = seq_index[:DEFINES.max_sequence_length - 1] + [dictionary[END]]
        else:
            seq_index += [dictionary[END]]
            
        # seq의 길이를 저장
        seq_len.append(len(seq_index))
        
        # DEFINES.max_sequence_length 길이보다 작은 경우 PAD 값을 추가 (padding)
        seq_index += (DEFINES.max_sequence_length - len(seq_index)) * [dictionary[PAD]]
        
        # 인덱스화 되어 있는 값은 seq_input_index에 추가
        seq_input_index.append(seq_index)
   
    return np.asarray(seq_input_index)

"""
배치 파트 데이터를 만드는 
"""

# input과 output dictionary를 만드는 함수
def in_out_dict(input, output, target):
    features = {"input": input, "output": output}
    return features, target


# 학습에 들어가 배치 데이터를 만드는 함수
def train_input_fn(train_input_enc, train_input_dec, train_target_dec, batch_size):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # train_input_enc, train_output_dec, train_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_input_dec, train_target_dec))
    # 전체 데이터를 썩는다.
    dataset = dataset.shuffle(buffer_size=len(train_input_enc))
    # 배치 인자 값이 없다면  에러를 발생 시킨다.
    assert batch_size is not None, "train batchSize must not be None"
    # from_tensor_slices를 통해 나눈것을
    # 배치크기 만큼 묶어 준다.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 데이터 각 요소에 대해서 in_out_dict 함수를
    # 통해서 요소를 변환하여 맵으로 구성한다.
    dataset = dataset.map(in_out_dict)
    # repeat()함수에 원하는 에포크 수를 넣을수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터 된다.
    dataset = dataset.repeat()
    # make_one_shot_iterator를 통해 이터레이터를
    # 만들어 준다.
    iterator = dataset.make_one_shot_iterator()
    # 이터레이터를 통해 다음 항목의 텐서
    # 개체를 넘겨준다.
    return iterator.get_next()


# 평가에 들어가 배치 데이터를 만드는 함수
def eval_input_fn(eval_input_enc, eval_input_dec, eval_target_dec, batch_size):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # eval_input_enc, eval_input_dec, eval_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_input_dec, eval_target_dec))
    # 전체 데이터를 섞는다.
    dataset = dataset.shuffle(buffer_size=len(eval_input_enc))
    # 배치 인자 값이 없다면  에러를 발생 시킨다.
    assert batch_size is not None, "eval batchSize must not be None"
    # from_tensor_slices를 통해 나눈것을
    # 배치크기 만큼 묶어 준다.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 데이터 각 요소에 대해서 in_out_dict 함수를
    # 통해서 요소를 변환하여 맵으로 구성한다.
    dataset = dataset.map(in_out_dict)
    # repeat()함수에 원하는 에포크 수를 넣을수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터 된다.
    # 평가이므로 1회만 동작 시킨다.
    dataset = dataset.repeat(1)
    # make_one_shot_iterator를 통해
    # 이터레이터를 만들어 준다.
    iterator = dataset.make_one_shot_iterator()
    # 이터레이터를 통해 다음 항목의
    # 텐서 개체를 넘겨준다.
    return iterator.get_next()

"""
단어 사전을 만드는 파트

load_voc(): 단어 사전 vocabularyData.voc를 생성하고 단어와 인덱스 관계를 출력

data_path에서 csv 파일 데이터 읽어오기 ->
읽어온 데이터 노이즈 캔슬링 ->
데이터 토큰화 ->
모든 토큰을 set을 통하여 중복없는 토큰 list 생성 ->
패딩 데이터 추가 ->
vocabularyData.voc 저장 ->
make_voc를 사용하여 각 토큰에 해당되는 인덱스와 인덱스에 해당되는 토큰 데이터 생성

output: {token1:index1, token2:index2,...}, {index1:token1, index2:token2,...}, length of voc
"""

def load_voc(token_train_q, token_train_a):
    # 사전을 담을 배열 준비한다.
    voc_list = []
    # 사전을 구성한 후 파일로 저장 진행한다.
    # 그 파일의 존재 유무를 확인한다.
    if (not (os.path.exists(DEFINES.vocabulary_path))):
       
        words = []
        # 질문과 답변을 extend을
        # 통해서 구조가 없는 배열로 만든다.
        words.extend(token_train_q)
        words.extend(token_train_a)
        
        s = set()
        for word in words:
            for token in word:
                s.add(token)
        
        words = list(s)
       
        words[:0] = MARKER
        
        # 사전을 리스트로 만들었으니 이 내용을
        # 사전 파일을 만들어 넣는다.
        with open(DEFINES.vocabulary_path, 'w', encoding='utf-8') as voc_file:
            for word in words:
                voc_file.write(word + '\n')

    # 사전 파일이 존재하면 여기에서
    # 그 파일을 불러서 배열에 넣어 준다.
    with open(DEFINES.vocabulary_path, 'r', encoding='utf-8') as voc_file:
        for line in voc_file:
            voc_list.append(line.strip())

    # make() 함수를 사용하여 dictionary 형태의 char2idx, idx2char 저장
    char2idx, idx2char = make_voc(voc_list)
    
    return char2idx, idx2char, len(char2idx)

"""
make_voc(voc_list): 사전 리스트를 받아 인덱스와 토큰의 dictionary를 생성

output: {token1:index1, token2:index2,...}, {index1:token1, index2:token2,...}
"""

def make_voc(voc_list):
    # 리스트를 키가 단어이고 값이 인덱스인
    # 딕셔너리를 만든다.
    char2idx = {char: idx for idx, char in enumerate(voc_list)}
    # 리스트를 키가 인덱스이고 값이 단어인
    # 딕셔너리를 만든다.
    idx2char = {idx: char for idx, char in enumerate(voc_list)}
    # 두개의 딕셔너리를 넘겨 준다.
    return char2idx, idx2char

"""
pred_next_string(value, dictionary): 예측용 단어 인덱스를 문장으로 변환하는 함수
input:
value: 텍스트 문장들 데이터
dictionary: 값이 인덱스인 단어 사전

단어 사전을 이용하여 각 인덱스의 해당되는 단어로 문장 형성->
패딩 값은 스페이스 처리 ->
END가 들어가면 출력완료 값 = True

output: 변환된 문장, 출력완료 boolean값
"""

def pred_next_string(value, dictionary):
    # 텍스트 문장을 보관할 배열을 선언한다.
    sentence_string = []

    # 인덱스 배열 하나를 꺼내서 v에 넘겨준다.
    for token in value:
        # 딕셔너리에 있는 단어로 변경해서 배열에 담는다.
        sentence_string = [dictionary[index] for index in token['indexs']]
    print(sentence_string)
    answer = ""
# 패딩값과 엔드값이 담겨 있으므로 패딩은 모두 스페이스 처리 한다.
    for word in sentence_string:
        if word not in PAD and word not in END:
            answer += word
            answer += " "

    print(answer)
    return answer

def main(self):
    char2idx, idx2char, vocabularyLength = loadVocabulary()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
