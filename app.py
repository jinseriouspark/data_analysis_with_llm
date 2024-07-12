from langchain_community.chat_models import ChatOllama
import streamlit as st
import pandas as pd

gemma = "gemma:7b-instruct"

with st.sidebar:
    ngrok_url = st.text_input("Ngrok URL", key="ngrok_url")

st.title("LLM Dual Analysis")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "분석하고자 하는 데이터셋을 업로드해주세요."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


def get_response(model, prompt, url):
    model_gemma = ChatOllama(model=model, temperature=0, base_url=url, verbose=True)
    return model_gemma.stream(prompt)


# 문서 분석 및 self-discovery 질문 생성
def analyze_document_and_generate_questions(document_text):
    # Ollama 모델을 사용해 문서 내용을 분석하고 self-discovery 질문을 생성
    new_prompt = f"""주어진 문서를 살펴보고, 제목 및 데이터셋 현황을 정리해주세요.
            한글 답변 부탁해. only korean answer please 
            document_text :\n\n{document_text}\n\n질문:"""
    response = st.write_stream(
        get_response(model=gemma, prompt=new_prompt, url=ngrok_url)
    )
    st.session_state.messages.append({"role": gemma, "content": response})


# 문서 분석 및 self-discovery 질문 생성: select
def select(dataframe, base_task_39):
    # Ollama 모델을 사용해 문서 내용을 분석하고 self-discovery 질문을 생성
    select_prompt = f"""
                  유저의 dataframe 데이터를 기반으로, 유저의 질문을 풀기 위한 task 5개를 선택하여라. 
                  task 5개는 아래 base_task_39 중에서 선택해야 한다. 

                  dataframe
                  {dataframe}

                  base_task_39
                  {base_task_39}

                  선택된 task 5: 
                  """
    response = st.write_stream(
        get_response(model=gemma, prompt=select_prompt, url=ngrok_url)
    )
    st.session_state.messages.append({"role": gemma, "content": response})


# adapt : 선택된 모듈을 작업 과제에 맞게 재구성
def adapt(dataframe, user_question, selected_task5):
    # Ollama 모델을 사용해 문서 내용을 분석하고 self-discovery 질문을 생성
    adapt_prompt = f"""
                  유저의 dataframe 데이터를 기반으로, 유저의 질문을 풀기 위한 task 5개가 선택되었다.
                  유저의 질문을 풀기 위해 더 구체적으로 task 5개를 변형하여라.

                  dataframe
                  {dataframe}

                  유저의 질문 
                  {user_question}

                  selected_task5
                  {selected_task5}

                  예시:
                    - selected_task 1 : 33 What kinds of solution typically are produced for this kind of problem specification?
                    - 변형된 task 1 : '상위 20% 유저와 하위 80%유저의 특징이 뚜렷하게 구분되지 않는 카테고리'가 많을 경우, 어떠한 솔루션들이 이러한 문제를 해결할 수 있을까?

                  변형된 task 5: 
                  """
    response = st.write_stream(
        get_response(model=gemma, prompt=adapt_prompt, url=ngrok_url)
    )
    st.session_state.messages.append({"role": gemma, "content": response})


def implement(dataframe, user_question, transformed_task5):
    # Ollama 모델을 사용해 문서 내용을 분석하고 self-discovery 질문을 생성
    implement_prompt = f"""
                  유저의 dataframe 데이터를 기반으로, 변형된 task 5개를 만들었다. 
                  해당 테스크를 차례대로 적용하면서, 유저의 질문에 대한 해결책을 제시해주세요. 
                  예시는 보여주지 말고 변형된 task과 해결책만 보여주세요.

                  dataframe
                  {dataframe}

                  유저의 질문 
                  {user_question}

                  변형된 task 5
                  {transformed_task5}

                  예시:
                    - task 1 :'상위 20% 유저와 하위 80%유저의 특징이 뚜렷하게 구분되지 않는 카테고리'가 많을 경우, 어떠한 솔루션들이 이러한 문제를 해결할 수 있을까?
                    - 해결책 1 : 두 그룹간의 특징이 뚜렷하게 구분되지 않는 이유는 카테고리 코드 A01B02때문입니다. 그러한 모습은 A001 전반적으로 드러납니다. 
                        페르소나 속 두 그룹을 구분짓는 유일한 카테고리는 A04B02 이며, 해당 카테고리는 '여성이 많고 남성이 적은' 특징을 보입니다. 
                        '여성과 남성' 데모 비중을 뚜렷하게 구분짓는 카테고리를 다시 한번 수정해서 분석 주제로 삼고, 분석 결과를 다시 살펴보세요.

                  해결책: 
                  """
    response = st.write_stream(
        get_response(model=gemma, prompt=implement_prompt, url=ngrok_url)
    )
    st.session_state.messages.append({"role": gemma, "content": response})


# 문서 업로드
uploaded_file = st.sidebar.file_uploader(
    "Upload a document", type=["txt", "pdf", "docx", "csv"]
)

# self-discover
with open("/content/base_task_39.txt", "r") as f:
    base_task_39 = f.read()

#


if uploaded_file is not None and st.sidebar.button("문서 분석", key="docu_analysis"):
    if uploaded_file.type == "text/plain":
        document_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfFileReader(uploaded_file)
        document_text = "\n".join(
            [reader.getPage(i).extract_text() for i in range(reader.numPages)]
        )
    elif (
        uploaded_file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        doc = Document(uploaded_file)
        document_text = "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        document_text = df.to_string(index=False)
    st.session_state.document_text = document_text

    questions = analyze_document_and_generate_questions(document_text)
    st.session_state.questions = questions

    selected_task5 = select(document_text, base_task_39)
    st.session_state.selected_task5 = selected_task5

    transformed_task5 = adapt(document_text, questions, selected_task5)
    st.session_state.transformed_task5 = transformed_task5

    implement_answer = implement(document_text, questions, transformed_task5)

if prompt := st.chat_input():
    if not ngrok_url:
        st.info("ngrok url 을 넣어주세요")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 이전의 세션 메시지를 기반으로 Gemma와의 대화에 추가
    previous_messages = st.session_state.messages[-3:]  # 이전 3개의 메시지 가져오기
    prompt_with_previous = f"{prompt}\n\n이전 질문 및 응답:\n\n"
    for i, msg in enumerate(previous_messages, start=1):
        prompt_with_previous += f"{i}. 질문: {msg['content']}\n   응답: {msg['role']}\n"

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 호출 및 전처리
    with st.chat_message(gemma):

        with st.spinner("대답중 ..."):
            response = st.write_stream(
                get_response(model=gemma, prompt=prompt, url=ngrok_url)
            )
    st.session_state.messages.append({"role": gemma, "content": response})
