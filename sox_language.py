import warnings

# For LLM and STT
from gpt4all import GPT4All


##### Send model work to this device...
if torch.cuda.is_available():
    torch_dtype = torch.float16 
    worker_device = "cuda:0"
else:
    torch_dtype = torch.float32
    worker_device="cpu"

#######################################################################################
# Name: LMInferenceProcessManager
#
# Purpose:  Encapsulates loading/managing LLM state as well as executing inference.
# 
# Caveats:  Currently single-language.  This will need a factory with specializations
#  per-language and a base-class/generic method for Infer in the long-term with some
# pickle-able configuration to allow users to use arbitrary language models.
#
#######################################################################################

class InferenceManager:

    def __init__(self, language):
        self.target_language = language
        self.language_model = self.getModelForLanguage(self.target_language)


    def getModelForLanguage(self, language):
        if "en" not in language:
            raise Exception("unsupported language for the language model subsystem");
        else:
            self.model_name = "gpt4all-falcon-newbpe-q4_0.gguf"
            self.max_tokens = 400
            self.format_string = "### Instruction:\n{request}\n### Response:\n"
            return GPT4All(self.model_name, device='gpu')


    def Infer(self, textRequest):
        formattedQuery = self.format_string.format(request = textRequest)
        output = self.language_model.generate(formattedQuery, max_tokens=self.max_tokens)
        return output
