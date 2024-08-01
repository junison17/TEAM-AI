import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from tavily import TavilyClient
from openai import OpenAI
import time
import base64
import tenacity
from openai import OpenAIError

# API 키를 st.secrets에서 가져옵니다
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

# OpenAI 클라이언트 설정
client = OpenAI(api_key=OPENAI_API_KEY)

# SerpAPI 설정
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

# Tavily 클라이언트 설정
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# 도구 설정
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="유용한 인터넷 검색 도구입니다. 최신 정보나 사실 확인이 필요할 때 사용하세요."
    ),
    Tool(
        name="Tavily Search",
        func=lambda query: tavily_client.search(query=query).get("results", []),
        description="Tavily API를 사용한 고급 검색 도구입니다. 심층적인 정보 검색이 필요할 때 사용하세요."
    )
]

# CrewAI용 OpenAI 함수 정의 (재시도 로직 강화)
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(OpenAIError),
    before_sleep=lambda retry_state: st.warning(f"API 오류 발생. {retry_state.attempt}/5 재시도 중...")
)
def openai_function(prompt):
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10000,
            stream=True,
        )
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
                yield response
    except OpenAIError as e:
        st.error(f"OpenAI API 오류: {str(e)}. 잠시 후 다시 시도합니다.")
        raise
    except Exception as e:
        st.error(f"예기치 못한 오류 발생: {str(e)}. 관리자에게 문의하세요.")
        raise

# 에이전트 정의
researcher = Agent(
    role='연구원',
    goal='주어진 주제에 대해 철저하고 정확한 연구를 수행합니다.',
    backstory='당신은 다양한 주제에 대해 깊이 있는 지식을 가진 숙련된 연구원입니다.',
    allow_delegation=False,
    verbose=True,
    tools=tools
)

critic = Agent(
    role='비평가',
    goal='연구 결과를 객관적으로 분석하고 개선점을 제시합니다.',
    backstory='당신은 날카로운 분석력과 비평 능력을 갖춘 전문가입니다.',
    allow_delegation=False,
    verbose=True,
    tools=tools
)

manager = Agent(
    role='선임매니저',
    goal='팀의 작업을 조율하고 최종 결과물을 승인합니다.',
    backstory='당신은 풍부한 경험을 가진 관리자로, 팀의 성과를 극대화하는 것이 목표입니다.',
    allow_delegation=False,
    verbose=True,
    tools=tools
)

ceo = Agent(
    role='대표이사',
    goal='회사의 전략적 방향을 설정하고 최종 의사결정을 내립니다.',
    backstory='당신은 회사의 대표이사로, 팀의 연구 결과를 바탕으로 중요한 의사결정을 내립니다.',
    allow_delegation=False,
    verbose=True,
    tools=tools
)

def run_meeting(user_prompt):
    try:
        st.write("\n--- 회의 시작 ---\n")

        research_task = Task(
            description=f"다음 주제에 대해 심도 있는 연구를 수행하세요: {user_prompt}",
            agent=researcher
        )

        critic_task = Task(
            description="연구 결과를 철저히 검토하고 구체적인 개선점을 제안하세요.",
            agent=critic
        )

        manager_task = Task(
            description="연구 결과와 비평을 심도 있게 검토하고 팀의 다음 단계를 제시하세요.",
            agent=manager
        )

        crew = Crew(
            agents=[researcher, critic, manager],
            tasks=[research_task, critic_task, manager_task],
            process=Process.sequential
        )

        result = crew.kickoff()

        st.write("팀의 결과:")
        output_placeholder = st.empty()
        for chunk in openai_function(result):
            output_placeholder.write(chunk)

        st.write("\n--- 최종 보고 ---\n")
        final_task = Task(
            description=f"팀의 회의 결과를 종합적으로 검토하고 최종 전략적 방향을 제시하세요:\n{result}",
            agent=ceo
        )

        final_crew = Crew(
            agents=[ceo],
            tasks=[final_task],
            process=Process.sequential
        )

        final_result = final_crew.kickoff()
        st.write("대표이사의 최종 의견:")
        ceo_placeholder = st.empty()
        for chunk in openai_function(final_result):
            ceo_placeholder.write(chunk)

        return result, final_result
    except Exception as e:
        st.error(f"회의 진행 중 오류가 발생했습니다: {str(e)}. 잠시 후 다시 시도해주세요.")
        return None, None

def get_table_download_link(text):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="meeting_minutes.txt">회의록 다운로드</a>'

def main():
    st.title("CrewAI 팀 회의 시뮬레이션")

    user_prompt = st.text_area("연구 주제를 입력하세요:", height=100)

    if 'meeting_started' not in st.session_state:
        st.session_state.meeting_started = False

    if 'meeting_finished' not in st.session_state:
        st.session_state.meeting_finished = False

    if 'meeting_result' not in st.session_state:
        st.session_state.meeting_result = None

    if 'ceo_result' not in st.session_state:
        st.session_state.ceo_result = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("회의 시작"):
            if user_prompt:
                st.session_state.meeting_started = True
                st.session_state.meeting_finished = False
                st.session_state.meeting_result = None
                st.session_state.ceo_result = None
                
                with st.spinner("팀원이 회의중입니다. 회의가 종료되면 보고 드리겠습니다..."):
                    st.session_state.meeting_result, st.session_state.ceo_result = run_meeting(user_prompt)
                
                st.session_state.meeting_finished = True
                if st.session_state.meeting_result is not None and st.session_state.ceo_result is not None:
                    st.success("회의가 종료되었습니다.")
                else:
                    st.error("회의 진행 중 오류가 발생했습니다. 다시 시도해 주세요.")

    with col2:
        if st.button("회의 종료", disabled=not st.session_state.meeting_started):
            st.session_state.meeting_started = False
            st.session_state.meeting_finished = True
            st.warning("회의가 종료되었습니다.")

    if st.session_state.meeting_finished:
        if st.session_state.meeting_result is not None and st.session_state.ceo_result is not None:
            st.write("\n--- 최종 회의 결과 요약 ---\n")
            full_report = f"회의 결과:\n{st.session_state.meeting_result}\n\n대표이사의 최종 의견:\n{st.session_state.ceo_result}"
            
            st.write("회의 결과 요약:")
            st.write(st.session_state.meeting_result[:4000] + "...")

            st.write("\n대표이사의 최종 의견 요약:")
            st.write(st.session_state.ceo_result[:4000] + "...")

            st.markdown(get_table_download_link(full_report), unsafe_allow_html=True)

            # 사용자(대표이사)의 추가 질문 기능
            st.write("\n--- 추가 질문 ---\n")
            additional_question = st.text_input("팀원에게 추가 질문이 있다면 입력해주세요:")
            if st.button("질문하기"):
                if additional_question:
                    answer_task = Task(
                        description=f"대표이사의 다음 질문에 대해 팀을 대표하여 상세히 답변해주세요: {additional_question}\n컨텍스트 - 연구주제: {user_prompt}, 최종보고서: {st.session_state.meeting_result}",
                        agent=manager
                    )
                    answer_crew = Crew(
                        agents=[manager],
                        tasks=[answer_task],
                        process=Process.sequential
                    )
                    try:
                        answer_result = answer_crew.kickoff()
                        st.write("팀의 답변:")
                        answer_placeholder = st.empty()
                        for chunk in openai_function(answer_result):
                            answer_placeholder.write(chunk)
                    except Exception as e:
                        st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}. 잠시 후 다시 시도해주세요.")
        else:
            st.error("회의 결과를 불러올 수 없습니다. 다시 시도해 주세요.")

if __name__ == "__main__":
    main()