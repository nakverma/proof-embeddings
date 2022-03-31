from enum import Enum


class InvalidStates(Enum):
    INPUT_SYNTAX_ERROR = "Syntax error in input expression",
    INVALID_NEW_EXPR = "Invalid next step for current expression",
    INVALID_NEW_RULE = "Invalid rule applied to current expression",
    INCORRECT_RULE_EXPR = "Input expression not entailed by selected rule",
    UNKNOWN = "Default State: Error Unknown"


class InvalidExpressionException(Exception):

    def __init__(self, state):
        super().__init__()
        if state in InvalidStates:
            self.state = state
        else:
            self.state = InvalidStates.UNKNOWN

    def get_error_dict(self):
        return {
            "isValid": False,
            "isSolution": False,
            "errorCode": self.state.name,
            "errorMsg": self.state.value[0]
        }
