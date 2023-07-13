from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from get_env import load_env


def track_tokens_usage(llm, query):
    """直接使用大模型进行调用openai 接口"""
    with get_openai_callback() as cb:
        result = llm(query)
        # 以下为打印真实的消耗和具体费用
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    return result


def chain_tokens_usage(chain, query):
    """使用 chain 调用 openai 接口"""
    with get_openai_callback() as cb:
        result = chain.run(query)
        # 以下为打印真实的消耗和具体费用
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    return result


if __name__ == '__main__':

    llm = OpenAI(
        temperature=0,
        openai_api_key=load_env('OPENAI_API_KEY'),
        # openai_api_base='https://proxy1.one1.chat/v1/',
        model_name="gpt-3.5-turbo-16k"
    )
    # --------- 不具备上下文理解能的调用方式
    # while True:
    #     print("输入问题")
    #     ls = input()
    #     print("\n-----\n")
    #     print(track_tokens_usage(llm, ls))
    #     print("\n 回答结束 \n")

    # --------- 理解上下文理解能的调用方式
    """
    Memory: 表示基于内存记录文但得上下文内容
    ConversationBufferMemory :  保存为原始的问答内容和回答内容
    ConversationSummaryMemory:  会结合之前的回答生成新的摘要精心保存
    """
    # conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
    conversation = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))
    while True:
        print("输入问题")
        ls = input()
        print("\n-----\n")
        # 这里可以去掉空格,标点
        print(chain_tokens_usage(conversation, ls))
        print("***********************   打印具体的保存内容 *************************")
        print(conversation.memory.buffer)
        print("***********************   打印具体的保存内容 *************************")
        print("\n 回答结束 \n")


