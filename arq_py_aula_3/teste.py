from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel


#1º exercício

# def receive_input(str: str) -> str:
#     return str

# runnable = RunnablePassthrough(receive_input)

# resposta = runnable.invoke("Teste")

# print(resposta)


# runnable = RunnablePassthrough.assign(input_receive=lambda x:print(x["input"])) | RunnablePassthrough()

# resposta = runnable.invoke({"input": "CONCLUÍDO"})

# print(resposta)

#2º exercício

# def count_caracters(str:str) -> int:
#     return len(str)


# runnable1 = RunnablePassthrough()
# runnable2 = RunnableLambda(count_caracters)

# sequence = runnable1 | {
#     "num_caracters": runnable2
# }

# response = sequence.invoke("ola")
# print(response)

