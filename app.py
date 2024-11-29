# streamlit_app.py

import streamlit as st
import os
import openai
import json
#from dotenv import load_dotenv

# OpenAI API 키 설정
#load_dotenv()
# Initialize Streamlit session state for API key
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = ''

# Function to set OpenAI API key
def set_api_key():
    st.session_state['OPENAI_API_KEY'] = "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"
    st.session_state['password_submitted'] = True

# Display API key input form if not set
if not st.session_state.get('password_submitted', False):
    st.sidebar.header("Password Configuration")
    with st.sidebar.form("api_key_form"):
        st.write("Enter your Password:")
        st.text_input("password", key='api_key_input', type='password')
        submitted = st.form_submit_button("Submit")
        if submitted:
            if st.session_state['api_key_input']=="1234" and not st.session_state.get('password_submitted', False):
                set_api_key()
                st.success("Password submitted!")
                st.rerun()
            else: 
                st.error("Please enter a valid password.")

# If API key is not set, stop the execution
if not st.session_state.get('password_submitted', False):
    st.warning("Please enter your password in the sidebar.")
    st.stop()

# Set OpenAI API key from session state
openai.api_key = st.session_state['OPENAI_API_KEY']

# Retriever 클래스 정의
class Retriever:
    def __init__(self, test:str):
        self.test=test
        pass

    def get_plan(self, user_query: str):
        prompt = f"""
        The user is solving a programming test with the following content: "{self.test}"\n 
        While doing so, the user has asked the following question: "{user_query}"

        Create a long-term dialogue plan based on the user's question. The plan should help achieve the user's objective in a step-by-step format. Each step should include:
        - The first item should be an empty step with idx 0
        - The `objective` should reflect what the user wants to achieve in code generation
        - An `idx` field, starting from 1 for the first meaningful step
        - A specific action to take for each step, written to guide the conversation towards answering the question

        Example format:
        {{
            "objective": "A description of what the user wants to accomplish based on the query",
            "plan": [
                {{"idx": 0, "step": "빈 단계입니다."}},
                {{"idx": 1, "step": "Ask the user if order needs to be preserved when removing duplicates."}},
                {{"idx": 2, "step": "If order is unimportant, suggest using set() to remove duplicates."}},
                {{"idx": 3, "step": "If order is important, suggest using dict.fromkeys() to remove duplicates."}},
                {{"idx": 4, "step": "If the user presents a more complex case, explain OrderedDict or a for loop."}}
            ]
        }}

        Now, generate the plan based on the user query.
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Make sure to return in json format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7, 
                response_format={"type": "json_object"},
            )

            plan = response.choices[0].message.content
            return plan

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return {"error": "Failed to generate plan."}

# Generator 클래스 정의
class Generator:
    def __init__(self):
        pass

    def generator_prompt(self, user_query:str, objective:str, current_plan:str, step_idx:int,dialogue_history:list) -> list:
        instruction_prompt = {
            "role": "system",
            "content": (
                "You are a chatbot that assists the user with code writing based on a sequence of instructions."
                f" Your primary objective is: {objective}."
                " You will be guided by a detailed, step-by-step plan and informed about your current step to tailor your responses appropriately."
                f" For this interaction, you are on step {step_idx} of the plan: \"{current_plan[step_idx]['step']}\"."
                " When providing examples, write pseudocode in Korean."
                " Respond in Korean."
                " Always encourage the user to attempt the task independently and ask for clarification if they encounter difficulties."
            )
        }
        dialogue_prompt = []
        if dialogue_history:
            for dialogue in dialogue_history[-10:]:  # 최근 10개의 대화만 포함
                dialogue_prompt.append({"role": "user", "content": dialogue["user"]})
                dialogue_prompt.append({"role": "assistant", "content": dialogue["assistant"]})

        if user_query:
            dialogue_prompt.append({"role": "user", "content": user_query})

        dialogue_prompt.insert(0, instruction_prompt)
        return dialogue_prompt

    def get_answer(self, user_query: str, plan_json: dict, step:int=0, dialogue_history:list=[]):
        objective = plan_json["objective"]
        current_plan = plan_json["plan"]

        prompt = self.generator_prompt(user_query, objective, current_plan, step, dialogue_history)

        agent_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=500,
            temperature=0.7,
        )
        return agent_response.choices[0].message.content

# Checker 클래스 정의
class Checker:
    def __init__(self):
        pass

    def generate_prompt(self, user_query, step, final_step, dialogue_history, retriever_plan):
        plan = retriever_plan
        initial_prompt = f"""
            You are part of a helpful tutoring assistant for coding.

            The ultimate goal this student wants to solve through this conversation is: {plan["objective"]}
            The followings are plans to provide step-by-step instructions to the student: {plan["plan"]},
            and the student is currently working on step number {step} (which corresponds to the "idx" key of the plans.)

            Given these plans and the user's query, your task is to check which step should the user be in the next turn.
            Set the step number to 0 in the following cases:
            - If the provided explanation does not match user's intent at all so that the entire plan needs to be modified
            - When the user asks for new content that is completely different from the configured ultimate goal
            For example, when the user says "This is quite different from what I asked. I want to know how to index a dataframe, not a list.",
            or "Okay, then I'll ask you another section. How can I arrange the dataframe?", the step is 0.

            Else,
            - Consider and compare the content of the user query and each step. Return the step corresponding to the user query.
            - If the student seems to have questions left before proceeding to the next step, keep the step.
            - If the query doesn't perfectly match with any of the plans but the student seems to have fully understood the current step, increase the step to (step+1).
            - If the student has reached the ultimate goal and there is no need to continue the conversation, increase the step to {final_step+1}.
            For example, when the user says "I think I've got it. Thank you!", the step should be {final_step + 1}.

            Answer only by the number of the step.
        """

        prompt = [{"role": "system", "content": initial_prompt}]
        if dialogue_history:
            prompt.append({"role": "user", "content": dialogue_history[-1]["user"]})
            prompt.append({"role": "assistant", "content": dialogue_history[-1]["assistant"]})
        prompt.append({"role": "user", "content": user_query})
        return prompt

    def check(self, user_query, step, final_step, dialogue_history, retriever_plan):
        prompt = self.generate_prompt(user_query, step, final_step, dialogue_history, retriever_plan)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=10,
            temperature=0.0,
        )

        checked_step = response.choices[0].message.content.strip()
        try:
            return int(checked_step)
        except ValueError:
            return step  # 변환 실패 시 현재 단계를 유지

#####파일 관련

def load_tests_from_json(file_path):
    titles = ["선택 없음"]  # "선택 없음" 옵션 추가
    tests = {"선택 없음": []}
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for instance in data:
        if "title" in instance:
            titles.append(instance["title"])
            tests[instance["title"]] = instance
    return titles, tests


###

# Streamlit 앱 초기화
def main():
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 200px !important; # Set the width to your desired value
            }
            .block-container {
                padding: 5rem 0rem; /* Adjust padding (top/bottom, left/right) */
            }
            .divider {
                border-left: 1px solid #ccc;
                height: 100%;
                position: absolute;
                left: 50%;
            }
            .container {
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
        .chat-container .chat-input {
            margin-top: auto; /* Pushes input to the bottom */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<style>[data-testid="column"]:nth-child(2){background-color: lightgrey;}</style>', unsafe_allow_html=True)

    test_save_path = "data.json"
    titles, tests = load_tests_from_json(test_save_path)

    if 'selected_test' not in st.session_state:
        st.session_state['selected_test'] = '선택 없음'
#    if 'basic_context' not in st.session_state:
#        st.session_state["basic_context"] = ""
    dir_name = st.session_state['selected_test']
    messages_save_path = f"results/{dir_name}/messages.json"


#    def update_articles():
#        st.session_state["tests"] = []
#        selected_test = st.session_state['selected_test']

#        if selected_test in tests and selected_test != "선택 없음":
#            text_content=tests[selected_test]["text"]
#            st.session_state["tests"].append(text_content)
    
    def reset_session_state():
        st.session_state.messages = []
        st.session_state.step = 1
        st.session_state.retriever_plan = None
        st.session_state.final_step = None

    with st.sidebar:
        st.header("다음 중 원하는 문제를 고르세요.")
        st.radio(
            "Choose a test:", titles, key="selected_test", on_change=reset_session_state
        )

    
    if st.session_state['selected_test']=="선택 없음":
        st.title("GPTeacher Assistant")
        st.write("문제를 골라주세요")
    else:
        title_info=tests[st.session_state['selected_test']]["title"]
        st.title("문제: "+title_info)
        
#        info_tab, chat_tab= st.tabs(["info", "chat"])
        col1, col2, col3 = st.columns([1,0.05,1])                
        with col1:
            text_info=tests[st.session_state['selected_test']]["text"]
            st.write(text_info)
       
        with col2:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        with col3:
            with st.container(height=450):
                st.write("코딩 질문에 도움을 주는 AI 어시스턴트입니다.")


                if "messages" not in st.session_state:
                    st.session_state.messages = []
                if "step" not in st.session_state:
                    st.session_state.step = 1
                if "retriever_plan" not in st.session_state:
                    st.session_state.retriever_plan = None
                if "final_step" not in st.session_state:
                    st.session_state.final_step = None

                retriever = Retriever(tests[st.session_state['selected_test']]["title"])
                generator = Generator()
                checker = Checker()

                # 이전 채팅 메시지 표시
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])


                # 사용자 입력 처리
                user_query = st.chat_input("질문을 입력하세요:", key="chat_input")
                if user_query:
                    # 사용자 메시지 표시
    #                with st.chat_message("user"):
    #                    st.markdown(user_query)
                    # 메시지 기록에 추가
                    st.session_state.messages.append({"role": "user", "content": user_query})

                    if st.session_state.retriever_plan is None or st.session_state.step == 0:
                        plan_str = retriever.get_plan(user_query)
                        try:
                            st.session_state.retriever_plan = json.loads(plan_str)
                        except json.JSONDecodeError:
                            st.error("계획을 파싱하는 데 실패했습니다. 다시 시도해주세요.")
                            return
                        st.session_state.final_step = len(st.session_state.retriever_plan["plan"]) - 1
                        st.session_state.step = 1

                    # 대화 기록 재구성
                    dialogue_history = []
                    messages = st.session_state.messages
                    i = 0
                    while i < len(messages):
                        if messages[i]["role"] == "user":
                            user_msg = messages[i]["content"]
                            assistant_msg = ""
                            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                                assistant_msg = messages[i + 1]["content"]
                                i += 1
                            dialogue_history.append({"user": user_msg, "assistant": assistant_msg})
                        i += 1

                    # 답변 생성
                    answer = generator.get_answer(
                        user_query,
                        st.session_state.retriever_plan,
                        st.session_state.step,
                        dialogue_history,
                    )

                    # 어시스턴트의 답변 표시
    #                with st.chat_message("assistant"):
    #                    st.markdown(answer)
                    # 메시지 기록에 추가
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                    # 다음 단계 확인
                    checker_decision = checker.check(
                        user_query,
                        st.session_state.step,
                        st.session_state.final_step,
                        dialogue_history,
                        st.session_state.retriever_plan,
                    )
                    st.session_state.step = checker_decision
                    print(f"체커 결정: {checker_decision}")

                    if st.session_state.step == 0:
                        st.session_state.retriever_plan = None

if __name__ == "__main__":
    main()
