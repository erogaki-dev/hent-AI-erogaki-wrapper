from AbstractHentAIErogakiWrapperError import AbstractHentAIErogakiWrapperError

class NoCensoredRegionsFoundError(AbstractHentAIErogakiWrapperError):
    def __init__(self, description):
        super().__init__(description)
