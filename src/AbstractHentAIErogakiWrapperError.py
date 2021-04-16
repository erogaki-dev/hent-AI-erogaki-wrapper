from erogaki_wrapper_shared_python.AbstractErogakiWrapperError import AbstractErogakiWrapperError

class AbstractHentAIErogakiWrapperError(AbstractErogakiWrapperError):
    def __init__(self, description):
        self.component = "hent-AI-erogaki-wrapper"
        super().__init__(description)
