from typing import Generic, TypeVar

T = TypeVar("T")


class ConstantRandomParamsGenerator(Generic[T]):
    def __init__(self, v: T):
        self.v = v

    def get_random_params(self, _: object) -> T:
        return self.v


def assert_equal(obj1: object, obj2: object) -> None:
    assert obj1 == obj2
    assert not (obj1 != obj2)
    assert hash(obj1) == hash(obj2)


def assert_not_equal(obj1: object, obj2: object) -> None:
    assert obj1 != obj2
    assert not (obj1 == obj2)
    assert hash(obj1) != hash(obj2)
