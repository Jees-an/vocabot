import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# ===== 1. 환경 설정 =====
load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

BASE_DIR = os.path.dirname(__file__)
save_dir = os.path.join(BASE_DIR, "embedding_store")

# ===== 2. 벡터스토어 로드 =====
textbook_db = FAISS.load_local(
    os.path.join(save_dir, "textbook_index"),
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
wordlist_db = FAISS.load_local(
    os.path.join(save_dir, "wordlist_index"),
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ===== 3. 페이지 설정 =====
st.set_page_config(page_title="📘 어휘 교육을 위한 AI 튜터", page_icon="🎓", layout="centered")

# ===== 4. 상단 디자인 =====
st.markdown("""
    <style>
    .main {
        background-color: #faf8f4;
        color: #333333;
        font-family: "NanumSquare", sans-serif;
    }
    .title-box {
        text-align: center;
        background-color: #ffe6c9;  /* 따뜻한 살구빛 배경 */
        padding: 35px 25px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .emoji {
        font-size: 38px;
        margin-bottom: 5px;
    }
    .title-box h1 {
        font-size: 34px;
        font-weight: 800;
        color: #4b2e05;  /* 진한 갈색: 따뜻하고 안정적인 대비 */
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 17px;
        color: #5b4636;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        background-color: #ffb347;  /* 밝은 오렌지 */
        color: #fff;
        font-size: 13px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .notice {
        font-size: 13px;
        color: #6e5537;
        font-style: italic;
        margin-top: 8px;
        margin-bottom: 10px;
    }
    .sponsor {
        font-size: 13px;
        color: #4b2e05;
        margin-top: 12px;
        line-height: 1.6;
        background-color: #fff8ef;
        padding: 10px 15px;
        border-radius: 10px;
        display: inline-block;
        border: 1px solid #f1d7b5;
    }
    .sponsor b {
        color: #4b2e05;
    }
    </style>

    <div class="title-box">
        <div class="emoji">📘🎓💬</div>
        <h1>어휘 교육을 위한 AI 튜터</h1>
        <div class="badge">Beta</div>
        <p class="subtitle">교과서 및 어휘 평정 목록 기반 질의응답</p>
        <p class="notice">※ 본 시스템은 연구 및 교육 실험용 베타 버전입니다. 일부 응답은 실제 교과 내용과 다를 수 있습니다.</p>
        <div class="sponsor">
            이 챗봇은 대구광역시교육청과 경북대학교 AI·디지털 융합 교육혁신 플랫폼 사업단에서 지원 받은<br>
            <b>&lt;LLMs, 교육용 말뭉치, 인간 교육 전문가의 협업을 통한 어휘 교육 방안 연구&gt;</b><br>
            (연구책임자: <b>연세대학교 남길임 교수</b>)의 일환으로 개발되었습니다.
        </div>
    </div>
""", unsafe_allow_html=True)

# ===== 5. 사이드바 =====
with st.sidebar:
    st.header("⚙️ 설정")
    if st.button("🧹 대화 초기화"):
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("**데이터 출처**")
    st.caption("📘 교과서: 공통국어 1·2(미래엔)")
    st.caption("🧾 어휘 목록: 연구 팀 자체 개발")
    st.markdown("---")
    st.markdown("**문의**")
    st.caption("💌 문의 메일: san@knu.ac.kr")

# ===== 6. 검색 함수 =====
def retrieve_combined(query: str, k: int = 3) -> str:
    docs_textbook = textbook_db.similarity_search(query, k=k)
    docs_wordlist = wordlist_db.similarity_search(query, k=k)
    all_docs = docs_textbook + docs_wordlist
    combined_context = "\n\n".join([doc.page_content for doc in all_docs])
    return combined_context

# ===== 7. 시스템 프롬프트 & 체인 =====
system_prompt = """
너는 '고등학교 공통국어 어휘 교육용 AI 튜터'이다.

- 사용자가 어휘를 물으면 wordlist_db와 textbook_db를 참고해 다음을 제시:
  1) 어휘의 뜻, 품사, 중요도, 표준국어대사전 등재 여부
  2) 교과서 예문
  3) 중요도 높을수록 자세히 설명
- 사용자가 "문제", "퀴즈", "문항" 등의 단어를 언급하면
  중요도 5 이상 어휘 3~5개로 객관식/단답형 문제를 생성하고 정답+해설을 함께 제시.
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{context}\n\n질문: {question}")
])

qa_chain = LLMChain(llm=llm, prompt=prompt)  # 메모리 제거

# ===== 8. 세션 상태 =====
if "history" not in st.session_state:
    st.session_state["history"] = []
if "quiz_mode" not in st.session_state:
    st.session_state["quiz_mode"] = False
if "quiz_data" not in st.session_state:
    st.session_state["quiz_data"] = []

# ===== 9. 입력 받기 =====
query = st.chat_input("특정 단어의 뜻을 물어 보거나, 공통국어에서 중요한 어휘 목록을 요청해 보세요!")

if query:
    # 문제 관련 키워드 포함되면 문제 생성
    if any(word in query for word in ["문제", "퀴즈", "문항"]):
        with st.spinner("문제 생성 중..."):
            context = retrieve_combined("고등학교 공통국어 어휘 문제")
            result = qa_chain.predict(
                context=context,
                question="중요도 5 이상 어휘로 3문항 생성"
            )
            st.session_state["quiz_mode"] = True
            st.session_state["quiz_data"] = result.split("\n\n")
            st.session_state["history"].append(("bot", "🧩 문제를 생성했습니다. 아래 문제를 풀어보세요!"))
    else:
        # 일반 어휘 질의응답
        with st.spinner("답변 생성 중..."):
            context = retrieve_combined(query)
            result = qa_chain.predict(context=context, question=query)
            st.session_state["history"].append(("user", query))
            st.session_state["history"].append(("bot", result))

# ===== 10. 대화 표시 =====
for role, msg in st.session_state["history"]:
    st.chat_message("user" if role == "user" else "assistant").write(msg)

# ===== 11. 문제 풀이 영역 =====
if st.session_state["quiz_mode"]:
    st.markdown("### 🧩 문제 풀이")
    for i, q in enumerate(st.session_state["quiz_data"], start=1):
        if not q.strip():
            continue
        st.markdown(f"**Q{i}.** {q}")
        answer = st.text_input(f"정답 입력 (문제 {i})", key=f"ans_{i}")
        if answer:
            st.success(f"✅ 입력한 답: {answer}")