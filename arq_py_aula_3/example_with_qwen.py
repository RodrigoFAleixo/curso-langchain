from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core
  # ainda compatível em várias versões recentes

MODEL_PATH = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)   

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    top_k=50,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    template="Resuma o texto a seguir em uma frase clara: {texto}",
    input_variables=["texto"]
)
chain = LLMChain(llm=llm, prompt=prompt)

input_text = "A IA tem transformado várias indústrias ao automatizar tarefas repetitivas e ampliar capacidades humanas."
print(chain.run(texto=input_text))
