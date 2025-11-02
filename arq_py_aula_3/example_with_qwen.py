from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL_PATH = "Qwen/Qwen3-1.7B"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Configuração de quantização em 4 bits
def _4bits():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16"
    )

quant_config = _4bits()

# Carregar modelo
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    quantization_config=quant_config
)

# Criar pipeline de geração de texto
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=3000
)

# Adaptar pipeline para LangChain
hf = HuggingFacePipeline(pipeline=pipe)

# Definir prompt
mensagens = [
    ("system", "Você é um poeta brasileiro famoso e escreve poemas de no máximo {n_versos} versos."),
    ("human", "Escreva para mim um poema sobre {assunto}."),
]

prompt_template = ChatPromptTemplate.from_messages(mensagens)

# Encadear: prompt → modelo → parser
chain = prompt_template | hf | StrOutputParser()

# Executar
resposta = chain.invoke({"n_versos": "10", "assunto": "navios"})

print(resposta)
