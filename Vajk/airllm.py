# %%
!pip install -U  airllm
# %%
from airllm import AutoModel
# %%
model = AutoModel.from_pretrained("01-ai/Yi-34B")#"garage-bAInd/Platypus2-7B")
