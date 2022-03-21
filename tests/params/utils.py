from typing import Generic, TypeVar

T = TypeVar("T")


class ConstantRandomParamsGenerator(Generic[T]):
    def __init__(self, v: T):
        self.v = v

    def get_random_params(self, _: object) -> T:
        return self.v
