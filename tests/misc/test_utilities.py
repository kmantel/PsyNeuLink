import collections
import numpy as np
import pytest

from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, prune_unused_args, copy_iterable_with_shared, ContentAddressableList


@pytest.mark.parametrize(
    'arr, expected',
    [
        ([[0], [0, 0]], np.array([np.array([0]), np.array([0, 0])], dtype=object)),
        # should test these but numpy cannot easily create an array from them
        # [np.ones((2,2)), np.zeros((2,1))]
        # [np.array([[0]]), np.array([[[ 1.,  1.,  1.], [ 1.,  1.,  1.]]])]

    ]
)
def test_convert_all_elements_to_np_array(arr, expected):
    converted = convert_all_elements_to_np_array(arr)

    # no current numpy methods can test this
    def check_equality_recursive(arr, expected):
        if (
            not isinstance(arr, collections.Iterable)
            or (isinstance(arr, np.ndarray) and arr.ndim == 0)
        ):
            assert arr == expected
        else:
            assert isinstance(expected, type(arr))
            assert len(arr) == len(expected)

            for i in range(len(arr)):
                check_equality_recursive(arr[i], expected[i])

    check_equality_recursive(converted, expected)


def f():
    pass


def g(a):
    pass


def h(a, b=None):
    pass


def i(b=None):
    pass


@pytest.mark.parametrize(
    'func, args, kwargs, expected_pruned_args, expected_pruned_kwargs', [
        (f, 1, {'a': 1}, [], {}),
        (g, 1, {'x': 1}, [1], {}),
        (g, None, {'a': 1}, [], {'a': 1}),
        (h, None, {'a': 1, 'b': 1, 'c': 1}, [], {'a': 1, 'b': 1}),
        (h, [1, 2, 3], None, [1], {}),
        (i, None, {'a': 1, 'b': 1, 'c': 1}, [], {'b': 1}),
        (i, [1, 2, 3], None, [], {}),
    ]
)
def test_prune_unused_args(func, args, kwargs, expected_pruned_args, expected_pruned_kwargs):
    pruned_args, pruned_kwargs = prune_unused_args(func, args, kwargs)

    assert pruned_args == expected_pruned_args
    assert pruned_kwargs == expected_pruned_kwargs


class Foo:
    name = 'Foo'


class Bar:
    name = 'Bar'


class Baz(Foo):
    pass


list_test_parameters = [
    (
        [Foo(), Bar(), 0, [0]],
        [True, False, True, [True]]
    ),
    (
        [Foo(), [Bar(), 0], 0, [0, 0]],
        [True, [False, True], True, [True, True]]
    ),
    (
        [[Foo(), Bar(), [Foo(), 0]]],
        [[True, False, [True, True]]],
    ),
    (
        [[Foo(), Bar(), (Foo(), 0)]],
        [[True, False, [True, True]]],
    ),
    (
        [(Foo(), Bar(), [Foo(), 0])],
        [[True, False, [True, True]]],
    ),
    (
        [
            ContentAddressableList(component_type=Foo, list=[Foo(), Baz()]),
            ContentAddressableList(component_type=Bar, list=[Bar()])
        ],
        [[True, True], [False]]
    ),
    (
        collections.deque([0, Foo(), Bar()]),
        [True, True, False]
    )
]


def tupleize(obj):
    try:
        return tuple([tupleize(x) for x in obj])
    except TypeError:
        return obj


# TODO: write/add tests for dicts
@pytest.mark.parametrize(
    'obj, identical_indices',
    [
        *list_test_parameters,
        *[(tupleize(case), indices) for case, indices in list_test_parameters]
    ]
)
def test_copy_iterable_with_shared_lists_mixed(obj, identical_indices):
    def compare(a, b, identical_indices, parent_indices=None):
        if parent_indices is None:
            parent_indices = []

        for i in range(len(a)):
            new_parent_indices = [*parent_indices, i]
            if isinstance(a[i], collections.Iterable) and not isinstance(a[i], str):
                assert a[i] is not b[i], f'{a[i]} and {b[i]} should not be identical (index ({new_parent_indices})'
                compare(a[i], b[i], identical_indices[i], new_parent_indices)
            else:
                id_str = '' if identical_indices[i] else ' not'
                assert (a[i] is b[i]) == identical_indices[i], f'{a[i]} and {b[i]} should{id_str} be identical (index ({new_parent_indices})'

    res = copy_iterable_with_shared(obj, shared_types=Foo)
    assert obj is not res
    compare(obj, res, identical_indices)
